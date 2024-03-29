from collections import Counter
import itertools as itools
import os
import statistics
from statistics import mode

from mahotas.features import haralick
import networkx as nx
import numpy as np
import pandas as pd
from scipy import signal, ndimage as ndi
from scipy.spatial.distance import euclidean
import scipy.stats as st
from skimage import segmentation, draw, filters, measure, io as skio, \
    morphology, exposure, feature, transform, util
from sklearn.neighbors import NearestNeighbors, KDTree
import tifffile as tif

from . import fluorescence_estimation as fe

active_contour = segmentation.active_contour


def mask_organoids(img, region, min_organoid_size=2500):
    """Processes transmission image in order to find organoids. It performs:
    1. Rescaling of intensity to float.
    2. inversion of intensities.
    3. Gamma adjustment of 5.
    4. filtered with sobel to find edges. If it is 3D, a maximum projection of
    sobel is given.
    5. Thresholded with Otsu.
    6. small holes and small objects are removed according to
    min_organoid_size.

    Parameters
    ----------
    img : np.array 2D or 3D.
        transmission image of specific timepoint.
    region : list of 4 elements
        bbox of the region were the organoid should be found.
    min_organoid_size : int, optional
        minimum area in pixels of a typical organoid. (default=1000)

    Returns
    -------
    mask : np.array(dtype=bool)
        boolean mask corresponding to the segmentation performed.
    processed_image : np.array 2D always
        Sobel filtered image of the given image or stack. If it was 3D, a
        maximum projection is returned."""

    processed_image = util.img_as_float(img)
    processed_image = util.invert(processed_image)
    processed_image = exposure.adjust_gamma(processed_image, 5)

    dimensions = len(img.shape)

    if dimensions == 3:
        pass

    elif dimensions == 2:
        img = img[np.newaxis, :]

    else:
        raise ValueError('Mask organoids works with 2 or 3 dimensions but %s '
                         'were given' % dimensions)

    processed_image = np.asarray([filters.sobel(this_processed_image)
                                  for this_processed_image
                                  in processed_image])
    processed_image = np.nanmax(processed_image, axis=0)

    if img.shape[0] > 1:
        threshold = filters.threshold_otsu(processed_image)
    elif img.shape[0] == 1:
        threshold = filters.threshold_otsu(processed_image) / 2
    else:
        raise ValueError('Something really strange happened with the length of'
                         ' image')

    mask = processed_image > threshold

    # We discard all foreground corresponding to other regions. At this point
    # it might allow us to discard objects at border of region.
    (xmin, xmax, ymin, ymax) = region
    mask[:int(ymin), :] = 0
    mask[int(ymax):, :] = 0
    mask[:, :int(xmin)] = 0
    mask[:, int(xmax):] = 0

    mask = morphology.binary_closing(mask, selem=morphology.disk(15))
    mask = morphology.remove_small_objects(mask, min_size=min_organoid_size)
    mask = ndi.morphology.binary_fill_holes(mask)

    return mask, processed_image


def get_filled_mask(mask):
    """Fills the interior of a masked object."""
    new_mask = morphology.binary_closing(mask, selem=morphology.disk(10))
    for region in measure.regionprops(new_mask.astype(int)):
        filled_area = region['filled_image']
        bbox = region['bbox']
        new_mask = np.zeros_like(new_mask)
        new_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = filled_area
    return new_mask


def get_filled_snake_from_mask(mask):
    """Tries to close large unclosed organoid masks and create an initial snake
    from them."""

    new_mask = get_filled_mask(mask)

    init_snake = mask_to_snake(new_mask)
    init_snake = sort_snake(init_snake)
    init_snake = np.asarray([init_snake[:, 1], init_snake[:, 0]]).T

    return init_snake


def get_init_snake(mask, region):
    """Generates an initial snake by running active contours from region to the
     borders of the mask."""
    mask = morphology.binary_dilation(mask, selem=morphology.disk(3))
    distance = ndi.distance_transform_edt(~mask)
    distance += mask * 1000
    init_snake = find_external(distance, region, mult=-100, gamma=0.001)
    return init_snake


def get_masked_img(img, mask):
    """Returns the image where every pixel outside the snake is zero."""
    masked_img = np.zeros_like(img)
    # mask = snake_to_mask(snake, img.shape)
    masked_img[np.nonzero(mask)] = img[np.nonzero(mask)]

    return masked_img


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
                           init_snake, alpha=0.015, beta=0.01, gamma=gamma*10,
                           w_line=mult*0.1, w_edge=10)
            
    # im = filters.gaussian(img, 2)

    # img = w_line*img + w_edge*edge[0]
    # alpha=0.01, beta=0.1,
    # w_line=0, w_edge=1, gamma=0.01,
    # bc='periodic', max_px_move=1.0,
    # max_iterations=2500, convergence=0.1
    # snake = active_contour(im,
    #                        snake, alpha=0.015, beta=0.01, gamma=gamma*100,
    #                        w_line=mult*0.1, w_edge=5)
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
            snks.append(sort_snake(np.asarray(draw.circle_perimeter(
                x, y, r, shape=img.shape)).T))
        
        # Swapped X and Y
        [rr, cc] = draw.circle_perimeter(y, x, r, shape=img.shape)
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
    x = snake[:, 0]
    y = snake[:, 1]
    points = np.c_[x, y]

    clf = NearestNeighbors(2).fit(points)
    G = clf.kneighbors_graph()

    T = nx.from_scipy_sparse_matrix(G)
    order = list(nx.dfs_preorder_nodes(T))

    xx = x[order]
    yy = y[order]

    return np.asarray([xx, yy]).T


def mask_to_snake(mask):
    seg = segmentation.find_boundaries(mask)
    out = np.nonzero(seg)
    return np.asarray(out)


def snake_to_mask(snake, shape):
    img = np.zeros(shape, 'uint8')
    snake = sort_border(snake)
    try:
        rr, cc = draw.polygon(snake[:, 1], snake[:, 0], shape)
        img[rr, cc] = 1
    except ValueError:
        print(snake.shape, shape)
    return img


def sort_border(border_coords, piece_length=50, max_distance=20,
                max_iterations=100):
    """Sorts the coordinates of the border of the mask.

    Parameters
    ----------
    border_coords : array
        Coordinates of the border of the mask
    piece_length : int
        Maximum length of the pieces that are allowed after first sort
    max_distance : int
        Maximum distance between points allowed after second sort
    max_iterations : int
        Maximum number of deletions allowed on second sort

    Returns
    -------
    sorted_points : numpy.ndarray 2D
        Sorted array of points; shape=(number of points, 2)
    """
    if len(border_coords) == 2:
        points = np.c_[border_coords[1], border_coords[0]]
    else:
        points = border_coords.copy()
    
    # FInd first point
    _, _, theta = polar_snake(points)
    first_point = np.argsort(theta)[0]

    # First sort with KD Tree
    sorted_points = kdtree_sorter(points, first_point)

    # Separating in pieces and resorting
    sorted_points = resort_by_pieces(sorted_points,
                                     length_threshold=piece_length)

    # Deleting the remaining points that are far away
    sorted_points = delete_far_points(sorted_points,
                                      distance_threshold=max_distance,
                                      max_iterations=max_iterations)

    # It may happen after sorting that it is going anti-clockwise so we have to
    # invert order
    if sorted_points[100, 1] > sorted_points[0, 1]:
        sorted_points = sorted_points[::-1]

    return sorted_points


def kdtree_sorter(points, first_point=0):
    """Sorts points using KD Tree starting from first_point (index of point)"""
    sorted_points = []
    remaining_points = points.copy()

    nearest_point_ind = first_point
    sorted_points.append(remaining_points[nearest_point_ind])
    remaining_points = np.delete(remaining_points, nearest_point_ind, axis=0)

    while len(remaining_points) > 0:
        #     print('Remaining points: %s' % len(remaining_points))
        tree = KDTree(remaining_points, leaf_size=2,
                      metric='euclidean')  # Create a distance tree
        dist, nearest_point_ind = tree.query(sorted_points[-1].reshape(1, -1),
                                             k=1)

        sorted_points.append(remaining_points[nearest_point_ind][0][0])
        remaining_points = np.delete(remaining_points, nearest_point_ind,
                                     axis=0)
    return np.asarray(sorted_points)


def calculate_distances(points):
    """Calculates euclidean distance between consecutive points"""
    distance = np.asarray(
        [euclidean(points[i], points[i + 1]) for i in range(len(points) - 1)])
    return distance


def resort_by_pieces(sorted_points, length_threshold=100):
    """Separates the ordered points into pieces where big jumps have been
    done. Short pieces are discarded and the remaining ones are reordered."""
    far_points = np.where(calculate_distances(sorted_points) > 6)[0]

    if len(far_points) == 0:
        print('There were no far points')
        return sorted_points

    pieces = []
    pieces.append(sorted_points[0:far_points[0] + 1])

    for i in range(len(far_points) - 1):
        pieces.append(sorted_points[far_points[i] + 1:far_points[i + 1] + 1])

    if len(far_points) == 1:
        pieces.append(sorted_points[far_points[0] + 1:])
    else:
        pieces.append(sorted_points[far_points[i + 1] + 1:])

    long_pieces = []
    for this_piece in pieces:
        if len(this_piece) > length_threshold:
            long_pieces.append(this_piece)
    pieces = long_pieces

    new_sorted_points = []
    new_sorted_points.extend(pieces[0])
    pieces.pop(0)

    while len(pieces) > 0:
        pieces_distance_first = [
            euclidean(new_sorted_points[-1], this_piece[0]) for this_piece in
            pieces]
        pieces_distance_last = [
            euclidean(new_sorted_points[-1], this_piece[-1]) for this_piece in
            pieces]

        if min(pieces_distance_first) <= min(pieces_distance_last):
            ind = np.argmin(pieces_distance_first)
            new_sorted_points.extend(pieces[ind])
            pieces.pop(ind)

        elif min(pieces_distance_first) > min(pieces_distance_last):
            print('invert')
            ind = np.argmin(pieces_distance_last)
            new_sorted_points.extend(pieces[ind][::-1])
            pieces.pop(ind)

    new_sorted_points = np.asarray(new_sorted_points)
    return new_sorted_points


def delete_far_points(sorted_points, distance_threshold=20, max_iterations=40):
    """Deletes every point that is farther than distance_threshold, unless more
    than max_iterations is required."""
    new_sorted_points = sorted_points.copy()
    distance = calculate_distances(new_sorted_points)
    iteration = 0
    while any(distance > distance_threshold):
        new_sorted_points = np.delete(new_sorted_points,
                                      np.where(distance > 20)[0], axis=0)
        distance = calculate_distances(new_sorted_points)

        if iteration > max_iterations:
            print('Too many points would have been deleted. Try other direction.')
            iteration = 0
            new_sorted_points = sorted_points.copy()
            distance = calculate_distances(new_sorted_points)

            while any(distance > 20):
                new_sorted_points = np.delete(new_sorted_points,
                                              np.where(distance > 20)[0] + 1,
                                              axis=0)
                distance = calculate_distances(new_sorted_points)
                if iteration > max_iterations:
                    print('Too many points in both directions.')
                    return sorted_points
                iteration += 1
            return new_sorted_points

        iteration += 1

    return new_sorted_points


def euclid_dist(t1, t2):
    "Calculates euclidean distance between vector and point."
    t1 = np.asarray(t1).T
    t2 = np.asarray(t2)
    return np.sqrt(((t1-t2)**2).sum(axis=1))


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


def get_hu_moments(image):
    """Returns the Hu Moments of an image."""
    m = measure.moments(image, order=3).T
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]

    mu = measure.moments_central(image, center=(cr, cc), order=3).T

    nu = measure.moments_normalized(mu, order=3)

    hu = measure.moments_hu(nu)

    return hu


def get_description(mask, descriptors=None):
    """Returns a dictionary with the chosen descriptors of the mask."""
    if descriptors is None:
        descriptors = ['area', 'centroid', 'convex_area', 'eccentricity',
                       'equivalent_diameter', 'euler_number', 'extent',
                       'major_axis_length', 'minor_axis_length', 'moments_hu',
                       'perimeter', 'solidity', ]
    description = {}
    for region in measure.regionprops(measure.label(mask)):
        for prop in descriptors:
            description[prop] = region[prop]
        return description
    return description  #sometimes there is no region


def get_texture_description(img, mask):
    """Returns a dictionary with the texture descriptors of the masked
    image."""
    description = {}
    #img = img.astype('uint8')  # TODO: test implicancies of reducing resolution
    masked_img = get_masked_img(img, mask)
    weighted_hu_moments = get_hu_moments(masked_img)
    for j in range(7):
        description['intensity_hu_moment_' + str(j + 1)] = weighted_hu_moments[
            j]

    try:
        haral = haralick(masked_img, ignore_zeros=True, return_mean=True)
    except ValueError:
        haral = np.asarray([np.nan] * 13)

    for j in range(13):
        description['haralick_' + str(j + 1)] = haral[j]

    return description


def generate_description(mask, trans, fluo, auto):
    """Generates a dictionary with the descriptors of the snake and image
    provided."""
    # mask = snake_to_mask(snake, trans.shape)
    description = get_description(mask)
    description.update(get_texture_description(trans, mask))
    trans, fluo, auto = fe.correct_stacks(trans, fluo, auto)
    description.update(fe.get_fluorescence_estimators(fluo, mask))

    return description


def best_haralick(z, harals, ax=None):
    """Fits the harals value from a single Haralick Moment and returns the best
    z found.

    Parameters
    ----------
    z : numpy.array
        1D Array of z values for the Haralick Features
    harals : numpy.array
        1D Array of one set of Haralick Features
    ax : matplotlib.Axes (optional, default=None)
        If given, a plot of values and fit is done

    Returns
    -------
    tuple
        (best z value, best Haralick estimation)
    """
    z = z.astype(float)
    idx = np.isfinite(z) & np.isfinite(harals)

    if np.sum(idx) == 0:
        return np.nan, np.nan

    if np.sum(idx) < 3:
        # if there are not enough points for quadratic fit
        x_ver = np.mean(z[idx])
        y_ver = np.mean(harals[idx])

        return x_ver, y_ver

    p = np.polyfit(z[idx], harals[idx], 2)
    poly = np.poly1d(p)

    x_ver = -p[1] / (2 * p[0])
    y_ver = poly(x_ver)
    if ax:
        ax.plot(z, harals, 'o')
        zs = np.arange(min(z) * 0.9, max(z) * 1.1, (max(z) - min(z)) / 30)
        poly = np.poly1d(p)
        ax.plot(zs, poly(zs))
        ax.scatter(x_ver, y_ver, color='r')

    return x_ver, y_ver


def best_z_plane(z_bests, z_min=0, z_max=6, z_best_prev=3):
    """Finds the best z_plane by voting between the result of each Haralick
    Feature. If it is a timelapse, z_best_prev can be used to untie a draw.

    Parameters
    ----------
    z_bests : list, tuple, numpy.array
        List of values of best z given by Haralick Features.
    z_min : float, int (optional, default=0)
        minimum possible value of z
    z_max : float, int (optional, default=6)
        maximum possible value of z
    z_best_prev : int, float (optional, default=3)
        Previous best z plane to untie a draw

    Returns
    -------
    best z value
    """
    z_bests = [np.nan if this < z_min else np.nan if this > z_max else this
               for this in z_bests]
    rounded = filter(np.isfinite, z_bests)
    rounded = [round(this) for this in rounded]
    rounded = np.asarray(rounded)

    if len(rounded) == 0:
        return z_best_prev

    try:
        z_best = mode(rounded)

    except statistics.StatisticsError:
        ind = np.argmin(abs(rounded - z_best_prev))
        z_best = rounded[ind]

    return z_best


def best_hu(z, hu_matrix, ax=None, z_best_prev=0):
    """Finds the best Hu Moments from a list of them got from different
    z-stacks. It uses the highest Hu Moments to discard every plane with high
    fluctuations and returns the plane that has less high fluctuations in every
    z plane.

    Parameters
    ----------
    z : list, tuple, numpy.array
        list of z planes given
    hu_matrix : numpy.array 2D
        Matrix containing each Hu moment for each z plane (z, Hu Moments)
    ax : matplotlib.Axes (optional, default=None)
        If given, a plot is returned showing the contest between planes
    z_best_prev : int, float (optional, default=0)
        Best z plane found in previous stack to untie a possible draw

    Returns
    -------
    tuple
        best z plane, list of Hu Moments of this plane
    """
    best_zs = []
    for i in range(3, 7):
        hus = hu_matrix[:, i]

        hus = st.zscore(hus)
        best_z = np.argmin(abs(hus))
        best_zs.append(z[best_z])

    best_zs = np.asarray(best_zs)
    try:
        best = mode(best_zs)
    except statistics.StatisticsError:
        ind = np.argmin(abs(best_zs - z_best_prev))
        best = best_zs[ind]

    if ax:
        ax.scatter(np.arange(3, 7), best_zs)
        ax.axhline(y=best)
        ax.set_title('best z: %s' % best)

    best_ind = np.where(z == best)[0][0]
    return best, hu_matrix[best_ind, :]


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


def segment_timepoint(tran, region):
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

    Returns
    -------
    results : dict
        Returns a dictionary with the results of the analysis. or a list of
        results if more than one z plane was analyzed.
        keys:
            z : int
                Z plane number
            init_snake : list
                List of coordinates of the initial contour
            external_snakes : list
                List of coordinates of the external contour
            # internal_snakes : list
            #     List of coordinates of the internal contour
            # lumen_snakes : list
            #     List of coordinates of the lumen contour obtained from
            #     fluorescence channel
            """

    if len(tran.shape) == 2:
        tran = tran[np.newaxis, :]
    elif len(tran.shape) != 3:
        raise ValueError('Dimension of timepoint is neither 2 or 3.')

    mask, processed_image = mask_organoids(tran, region)

    if not mask.any():
        print('no initial mask')
    # init_snake = get_init_snake(mask, region)

    results = {'z': [], 'external_snake': [], 'mask': []}
    for z, tran_z in enumerate(tran):
        this_tran = util.img_as_float(tran_z)
        segmented = segmentation.morphological_chan_vese(this_tran,
                                                         20,
                                                         init_level_set=mask,
                                                         smoothing=4,
                                                         lambda2=2)
        if not segmented.any():
            print('no segmented mask')

        segmented = morphology.binary_closing(segmented,
                                              selem=morphology.disk(10))
        segmented = ndi.morphology.binary_fill_holes(segmented)
        labeled = measure.label(segmented)
        max_area = 0
        label = 1
        solidity = 0
        for region in measure.regionprops(labeled):
            if region.area > max_area:
                max_area = region.area
                label = region.label
                solidity = region.solidity

        if solidity < 0.5:
            labeled = segmentation.morphological_chan_vese(this_tran,
                                                           20,
                                                           init_level_set=labeled == label,
                                                           smoothing=4,
                                                           lambda2=2)

        e_snk = mask_to_snake(labeled == label)

        results['z'].append([z])
        results['external_snake'].append([e_snk])
        results['mask'].append([labeled == label])

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

        result = segment_timepoint(tran0, fluo0, region)

        e_snks.append(result['e_snk'])
        i_snks.append(result['i_snk'])
        l_snks.append(result['l_snk'])

    return e_snks, i_snks, l_snks


def timepoint_to_df(params):
    """Analyzes a single timepoint and generates a small pandas DataFrame.

    Parameters
    ----------
    params : list
        ndx, key, filepath, fluo_filepath, auto_filepath, region

    Returns
    -------
    df : pandas DataFrame
        Small DataFrame with the results of a single timepoint analysis."""

    ndx, key, filepath, fluo_filepath, auto_filepath, region = params

    print('analyzing timepoint %s from file %s' % (ndx, filepath))

    tran_img = tif.TiffFile(str(filepath))
    fluo_img = tif.TiffFile(str(fluo_filepath))
    auto_img = tif.TiffFile(str(auto_filepath))

    tran = tran_img.asarray(key=key)
    fluo = fluo_img.asarray(key=key)
    fluo = util.img_as_float(fluo)
    auto = auto_img.asarray(key=key)
    auto = util.img_as_float(auto)

    if len(tran.shape) == 2:
        tran = tran[np.newaxis, :]
    elif len(tran.shape) != 3:
        raise ValueError('Dimension of timepoint is neither 2 or 3.')

    # Some saved images have a bit problem and they have 2^15 offset value
    if tran[0, 0, 0] >= 2**15:
        tran -= 2**15

    to_save = segment_timepoint(tran, region)

    dfs = []
    dict_keys = list(to_save.keys())
    for vals in zip(*to_save.values()):
        df = pd.DataFrame({this_key: this_val
                           for this_key, this_val in zip(dict_keys, vals)})

        # Generate a description of textures and Hu moments of the mask
        description = generate_description(df['mask'].values[0],
                                           tran[df['z'].values[0]],
                                           fluo[df['z'].values[0]],
                                           auto[df['z'].values[0]],)

        for prop in description.keys():
            if isinstance(description[prop], (tuple, list, np.ndarray)):
                df[prop] = [description[prop]]
            else:
                df[prop] = description[prop]

        df['tran_path'] = str(filepath)
        df['fluo_path'] = str(fluo_filepath)
        df['auto_path'] = str(auto_filepath)
        df['crop'] = [region]
        df['timepoint'] = ndx

        df.drop(columns='mask', inplace=True)
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    return dfs
        
            
def test_circle():
    emsk = np.zeros((1000, 1000), dtype='uint8')
    rr, cc = draw.circle(500, 500, 50, shape=(1000, 1000))
    emsk[rr, cc] = 1

    imsk = np.zeros((1000, 1000), dtype='uint8')
    rr, cc = draw.circle(500, 500, 20, shape=(1000, 1000))
    imsk[rr, cc] = 1

    return measure_thickness(imsk, emsk)
