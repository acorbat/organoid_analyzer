import os
import itertools as itools

import multiprocessing
import yaml

import pandas as pd
from scipy import signal
import numpy as np
from skimage import segmentation, draw, filters, measure, io as skio, \
    morphology, exposure, feature, transform, util
from skimage.filters import sobel, threshold_otsu

active_contour = segmentation.active_contour


def mask_organoids(img, min_organoid_size=1000):
    """Processes transmission image in order to find organoids. It performs:
    1. Rescaling of intensity to float.
    2. inversion of intensities.
    3. Gamma adjustment of 5.
    4. filtered with sobel to find edges.
    5. Thresholded with Otsu.
    6. small holes and small objects are removed according to
    min_organoid_size.

    Parameters
    ----------
    img : np.array
        transmission image of specific timepoint.
    min_organoid_size : int
        minimum area in pixels of a typical organoid.

    Returns
    -------
    mask : np.array(dtype=bool)
        boolean mask corresponding to the segmentation performed."""

    processed_image = util.img_as_float(img)
    processed_image = util.invert(processed_image)
    processed_image = exposure.adjust_gamma(processed_image, 5)
    processed_image = sobel(processed_image)
    threshold = threshold_otsu(processed_image)
    mask = processed_image > threshold
    mask = morphology.remove_small_holes(mask, area_threshold=min_organoid_size)
    mask = morphology.remove_small_objects(mask, min_size=min_organoid_size)

    return mask


def snake_from_extent(extents, shape):
    """Generate coordinates for snake from the coordinates of crop."""
    (xmin, xmax, ymin, ymax) = extents
    r = [xmin, xmin, xmax, xmax]
    c = [ymin, ymax, ymax, ymin]
    return np.asarray(draw.polygon_perimeter(r, c, shape=shape, clip=False)).T


def find_external(img, init_snake, mult=-1, gamma=0.0001):
    """Applies active contours to img with an initial init_snake."""
    # init_stake = snake or Rectangle extents: (xmin, xmax, ymin, ymax).
    
    if isinstance(init_snake, (tuple, list)):
        init_snake = snake_from_extent(init_snake, img.T.shape)
        
    img = img / np.max(img)
    
    im = filters.gaussian(img, 5)
    snake = active_contour(im,
                           init_snake, alpha=0.015, beta=10, gamma=gamma,
                           w_line=mult*0.1, w_edge=1)       
            
    # im = filters.gaussian(img, 2)

    # img = w_line*img + w_edge*edge[0]
    # alpha=0.01, beta=0.1,
    # w_line=0, w_edge=1, gamma=0.01,
    # bc='periodic', max_px_move=1.0,
    # max_iterations=2500, convergence=0.1
    # snake = active_contour(im,
    #                        snake, alpha=0.015, beta=10, gamma=gamma*100,
    #                        w_line=mult*0.1, w_edge=1)
    return snake


def nearest_opposites(angles):
    shifted = np.arctan2(np.sin(angles + np.pi), np.cos(angles + np.pi))
    out = np.zeros(angles.shape, 'int')
    for ndx, a in enumerate(angles):
        out[ndx] = np.argmin(np.abs(out-a))

    return out


def find_internal(img, external):
    
    img = img / np.max(img)
    
    center, lengths, thetas = polar_snake(external)
       
    opondx = nearest_opposites(thetas)
    radii = (lengths + lengths[opondx]) / 2
        
    hough_radii = np.arange(int(np.min(radii)*.75), int(np.min(radii)*1.1))
    img = exposure.rescale_intensity(filters.gaussian(img, 3),
                                     in_range='image', out_range='uint8')
    edges = feature.canny(img, sigma=3, low_threshold=10, high_threshold=50)

    # Detect two radii
    hough_res = transform.hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii,
                                                         total_num_peaks=3)
    cx = np.asarray([int(tmp) for tmp in cx])
    cy = np.asarray([int(tmp) for tmp in cy])
    radii = np.asarray([int(tmp) for tmp in radii])
    
    msk = snake_to_mask(external, img.shape)
    
    valid = []
    
    DEBUG = False
    if DEBUG:
        print(radii)
        print(cx)
        snks = []

    for ndx, (x, y, r) in enumerate(zip(cx, cy, radii)):
        if DEBUG:
            snks.append(sort_snake(np.asarray(draw.circle_perimeter(x, y, r)).T))
        
        # Swapped X and Y
        [rr, cc] = draw.circle_perimeter(y, x, r)
        valid.append(np.all(msk[rr, cc] == 1))
        
        if DEBUG:
            print(np.sum(msk[rr, cc])/len(rr))
    
    if DEBUG:
        from organoid_analyzer import visvis
        visvis.show_snakes(img, external, *snks)
       
    if len(valid):
        sel = np.nonzero(valid)
        cx = cx[sel]
        cy = cy[sel]
        radii = radii[sel]
    else:
        print('Warning: all circles were invalid')
    
    if len(radii):
        sel = np.argmax(radii)
        snake = np.asarray(draw.circle_perimeter(cx[sel], cy[sel],
                                                 radii[sel])).T
        return sort_snake(snake), edges
    else:
        print('Warning: no circles found')
        return np.copy(external), edges


def polar_snake(snake):
    center = np.mean(snake, axis=0)
    delta = snake - center
    theta = np.arctan2(delta[:, 1],  delta[:, 0])
    return center, np.linalg.norm(delta, axis=1), theta


def sort_snake(snake):
    _, _, theta = polar_snake(snake)
    ndxs = np.argsort(theta)
    return snake[ndxs, :]


def mask_to_snake(mask):
    seg = segmentation.find_boundaries(mask)
    out = np.nonzero(seg)
    return np.asarray(out).T


def snake_to_mask(snake, shape):
    rr, cc = draw.polygon(snake[:,1], snake[:,0], shape)
    img = np.zeros(shape, 'uint8')
    img[rr, cc] = 1
    return img


def _and_(*conds):
    out = conds[0]
    for current in conds[1:]:
        out = np.logical_and(out, current)
    return out


def line(r1, c1, r2, c2, shape, blank=None):
    im = np.zeros(shape, dtype='uint8')
    if r1 == r2:
        im[r1, :] = 1
    else:
        m = (c2 - c1) / (r2 - r1)
        rr = np.arange(0, im.shape[1], 1/10)
        cc = (rr - r1) * m + c1
        rr = np.round(rr).astype('int')
        cc = np.round(cc).astype('int')
        
        # remove duplicates cause we are subsampling rr
        rr, cc = zip(*set(zip(rr, cc)))
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        sel = _and_(cc >= 0, cc < im.shape[1], rr >= 0, rr < im.shape[0])
        im[rr[sel], cc[sel]] = 1
        
    if blank is not None:
        im[np.nonzero(blank)] = 0
    
    return np.nonzero(im)
                    
    
def measure_thickness(internal, external):
    # paso
    insk = mask_to_snake(internal)
    center, rho, theta = polar_snake(insk)
    out = []
    for ndx, the in enumerate(theta):
        rr, cc = line(int(center[0]), int(center[1]), int(insk[ndx, 0]),
                      int(insk[ndx, 1]),
                      internal.shape, external)
        dd1 = np.hypot(rr - insk[ndx, 0], cc - insk[ndx, 1])
        dd2 = np.hypot(rr - center[0], cc - center[1])
        dd1[dd1 > dd2] = np.inf
        n = np.argmin(dd1)
        out.append([the, rr[n], cc[n], dd1[n]])
    
    return np.asarray(out)  


def _make_mask_patterns(root_folder, number):
    i_pattern = os.path.join(root_folder, '{:03d}', 'YFP',
                             'YFP_Mask_{:03d}_%04d.tiff').format(number, number)
    e_pattern = os.path.join(root_folder, '{:03d}', 'Trans',
                             'Trans_Mask_%04d.tiff').format(number)
    return i_pattern, e_pattern


def _make_imgs_patterns(root_folder, number):
    i_pattern = os.path.join(root_folder, '{:03d}', 'YFP',
                             'YFP_Mask_{:03d}_%04d.tiff').format(number, number)
    e_pattern = os.path.join(root_folder, '{:03d}', 'Trans',
                             'Trans_Mask_%04d.tiff').format(number)
    return i_pattern, e_pattern


def make_mask_patterns(root_folder, number):
    i_pattern = os.path.join(root_folder, '{:03d}', 'YFP',
                             'YFP_Mask_{:03d}_%04d.tiff').format(number, number)
    e_pattern = os.path.join(root_folder, '{:03d}', 'Trans',
                             'Trans_Mask_%04d.tiff').format(number)
    return i_pattern, e_pattern


def make_imgs_patterns(root_folder, number):
    i_pattern = os.path.join(root_folder, '{:03d}', 'YFP',
                             'YFP_Mask_{:03d}_%04d.tiff').format(number, number)
    e_pattern = os.path.join(root_folder, '{:03d}', 'Trans',
                             'Trans_Mask_%04d.tiff').format(number)
    return i_pattern, e_pattern


def _iter_imgs(root_folder, number, make_patterns):
    n = 1
    ipat, epat = make_patterns(root_folder, number)
    while True:
        try:
            imsk = np.sum(skio.imread(ipat % n), axis=2)
            emsk = np.sum(skio.imread(epat % n), axis=2)
        except FileNotFoundError:
            break
            
        yield imsk, emsk
        n += 1 
    

def angular_mn_mx(nangles):
    angles = np.linspace(-np.pi, np.pi, nangles, endpoint=True)
    first_it, second_it = itools.tee(angles)
    next(second_it)
    yield from zip(first_it, second_it)
    

def stat_in_angle(angles, values, min_angle, max_angle, func):
    sel = np.logical_and(angles >= min_angle, angles < max_angle)
    return func(values[sel])


def angular_stats(angles, values, nangles, func):
    out = []
    for mn, mx in angular_mn_mx(nangles):
        out.append(stat_in_angle(angles, values, mn, mx, func))
    
    return out


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_bulges(thickness_out, nangles, thresh=None):
    st = []
    for part in thickness_out:
        st.append(angular_stats(part[:, 0], part[:, 3], nangles, np.median))
        
    st = np.stack(st).T
    # st.shape (# angles, # timepoints)
    
    if thresh is None:
        thresh = filters.threshold_yen(st)
        binary = st >= thresh  
    elif thresh == 'min_per_tp':
        binary = np.zeros(st.shape, 'uint8')
        for ndx in range(st.shape[1]):
            sig = st[:, ndx]
            
            avg = np.mean(rolling_window(sig, 3), -1)
            stdev = np.std(rolling_window(sig, 3), -1)
            sel = np.argmin(avg)
            binary[:, ndx] = sig > (avg[sel] + 4 * stdev[sel])
            
    elif thresh == 'yen_per_tp':
        binary = np.zeros(st.shape, 'uint8')
        for ndx in range(st.shape[1]):
            sig = st[:, ndx]
            th = filters.threshold_yen(sig)
            binary[:, ndx] = sig > th
 
    elif thresh == 'peak':
        binary = np.zeros(st.shape, 'uint8')

        for ndx in range(st.shape[1]):
            sig = st[:, ndx]
            
            # TODO scale by angles
            pks = signal.find_peaks_cwt(sig, np.linspace(5, 10))
            
            for pk in pks:
                binary[pk, ndx] = 1
                last = sig[pk]
                continue
                for nn in range(pk+1, st.shape[0]):
                    if abs(sig[nn] - last)/last < .1:
                        break
                    binary[nn, ndx] = 1
                    last = sig[nn]
                    
                last = sig[pk]
                for nn in range(pk-1, 0, -1):
                    if abs(sig[nn] - last)/last < .1:
                        break
                    binary[nn, ndx] = 1
                    last = sig[nn]         
                    
    elif thresh == 'split':
        binary = np.zeros(st.shape, 'uint8')

        for ndx in range(st.shape[1]):
            sig = st[:, ndx]
            
            # TODO scale by angles
            pks = signal.find_peaks_cwt(sig, np.linspace(5, 10))
            binary[pks, ndx] = 1
            
            pks = [0, ] + list(pks) + [len(pks), ]

            avg = np.mean(rolling_window(sig, 3), -1)
            stdev = np.std(rolling_window(sig, 3), -1)
            
            for pk1, pk2 in zip(pks[:-1], pks[1:]):
                sig0 = st[pk1:pk2, ndx]
                avg0 = avg[pk1:pk2]
                stdev0 = stdev[pk1:pk2]
                
                if not len(sig0):
                    continue               

                sel = np.argmin(avg0)              
                
                if avg0[sel] < min(sig0[0], sig0[-1]):
                    binary[pk1:pk2, ndx] = sig0 > (avg0[sel] + 2 * stdev0[sel])
                    
        binary = morphology.binary_opening(binary)     

    elif thresh == 'antipeak':
        binary = np.zeros(st.shape, 'uint8')
        for ndx in range(st.shape[1]):
            sig = st[:, ndx]
            # TODO scale by anges
            pks = signal.find_peaks_cwt(-sig, np.linspace(5, 10))

            for pk in pks:
                binary[pk, ndx] = 1
                last = sig[pk]
                for nn in range(pk, st.shape[0]):
                    if abs(sig[nn] - last)/last > 10:
                        break
                    binary[nn, ndx] = 1
                    last = sig[nn]
                    
                last = sig[pk]
                for nn in range(pk, 0, -1):
                    if abs(sig[nn] - last)/last > 10:
                        break
                    binary[nn, ndx] = 1
                    last = sig[nn]
            
    elif isinstance(thresh, (int, float)):
        binary = st >= thresh
      
    labeled_image = measure.label(binary)
    
    for ndx in range(st.shape[1]):
        if st[-1, ndx] > 0 and st[0, ndx] > 0 and st[-1, ndx] != st[0, ndx]:
            labeled_image[labeled_image == st[-1, ndx]] = st[0, ndx]
    
    return st, binary, labeled_image
    

def iter_masks(root_folder, number):
    for a, b in _iter_imgs(root_folder, number, make_mask_patterns):
        a[a > 0] = 1
        b[b > 0] = 1
        yield a, b
        
        
def iter_imgs(root_folder, number):
    for a, b in _iter_imgs(root_folder, number, make_mask_patterns):
        a[a > 0] = 1
        b[b > 0] = 1
        yield a, b
                

def create_thickness_out(internals, externals):
    return [measure_thickness(imsk, emsk)
            for imsk, emsk in zip(internals, externals)]


def create_bulge_masks(internals, externals, labeled_bulge):
    
    ub = np.unique(labeled_bulge)
    
    # labeled_bulge.shape (# angles, # timepoints)
    nangles = labeled_bulge.shape[0]
        
    # rows, columns, time, (ext, int, epi, epi no bulge, bulge 1, bulge 2 ...)
    out = None    
    
    for ndx, (imsk, emsk) in enumerate(zip(internals, externals)):
        insk = mask_to_snake(imsk)
        center, rho, theta = polar_snake(insk)

        if out is None:
            out = np.zeros((len(internals), 3 + len(ub),) + imsk.shape,
                           dtype='uint16')
                           
        epi_msk = emsk - imsk
        [rr, cc] = np.nonzero(epi_msk)
        theta = np.arctan2(cc - center[1], rr - center[0])
        
        out[ndx, 0, :, :] = emsk
        out[ndx, 1, :, :] = imsk
        out[ndx, 2, :, :] = epi_msk
                           
        for an_ndx, (mn, mx) in enumerate(angular_mn_mx(nangles)):
            sel = np.nonzero(np.logical_and(theta >= mn, theta < mx))
                                       
            out[ndx, 3 + labeled_bulge[an_ndx, ndx], rr[sel], cc[sel]] = 1
                           
    return out


def get_description(mask, descriptors=None):
    if descriptors is None:
        descriptors = ['area', 'centroid', 'convex_area', 'eccentricity',
                       'equivalent_diameter', 'euler_number', 'extent',
                       'major_axis_length', 'minor_axis_length', 'moments_hu',
                       'perimeter', 'solidity', ]
    description = {}
    for region in measure.regionprops(mask):
        for prop in descriptors:
            description[prop] = region[prop]
        return description


def protocol(stack, region):
    e_snks = []
    i_snks = []
    for ndx in range(stack.shape[0]):
        print(ndx)
        image0 = stack[ndx, :, :]
        
        e_snk = find_external(image0, region)
        i_snk, _ = find_internal(image0, e_snk)
        
        e_snks.append(e_snk)
        i_snks.append(i_snk)
    
    return e_snks, i_snks


def segment_timepoint(tran, fluo, region):
    """Analyzes a single pair of transmission and fluorescence timepoint, with
    the specified region of the desired organoid and returns a dictionary with
    the results.

    Parameters
    ----------
    tran : np.array
        transmission image of the organoids
    fluo : np.array
        fluorescence channel of the organoids
    region : list, tuple
        coordinates of the crop region

    Returns :
    -------
    results : dict
        Returns a dictionary with the results of the analysis.
        keys:
            external_snakes : list
                List of coordinates of the external contour
            internal_snakes : list
                List of coordinates of the internal contour
            lumen_snakes : list
                List of coordinates of the lumen contour obtained from
                fluorescence channel"""

    mask = mask_organoids(tran)
    e_snk = find_external(mask, region)
    i_snk, _ = find_internal(tran, e_snk)
    l_snk = find_external(fluo, region)

    results = {'external_snakes': [e_snk], 'internal_snakes': [i_snk],
               'lumen_snakes': [l_snk]}

    return results


def analyze_timeseries(stacks, region):
    """Analyzes a complete stack using region as crop coordinates and returns a
    list of snakes for each border found."""
    e_snks = []
    i_snks = []
    l_snks = []
    for ndx in range(stacks.shape[1]):
        print(ndx)
        tran0 = stacks[0, ndx, :, :]
        fluo0 = stacks[1, ndx, :, :]

        result = analyze_timepoint(tran0, fluo0, region)

        e_snks.append(result['e_snk'])
        i_snks.append(result['i_snk'])
        l_snks.append(result['l_snk'])

    return e_snks, i_snks, l_snks
        
            
def test_circle():
    emsk = np.zeros((1000, 1000), dtype='uint8')
    rr, cc = draw.circle(500, 500, 50, shape=(1000, 1000))
    emsk[rr, cc] = 1

    imsk = np.zeros((1000, 1000), dtype='uint8')
    rr, cc = draw.circle(500, 500, 20, shape=(1000, 1000))
    imsk[rr, cc] = 1

    return measure_thickness(imsk, emsk)
