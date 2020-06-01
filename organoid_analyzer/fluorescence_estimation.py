import numpy as np
from skimage import filters, morphology as morph

from img_manager import CorrectorArmy
from img_manager.correctors import ShiftCorrector, RollingBallCorrector, \
    BleedingCorrector
from . import morpho


def create_army():
    army = CorrectorArmy()
    army.add_channel('Trans')
    army.add_channel('Fluo')
    army.add_channel('Auto')
    army.add_channel('Red')

    # Add shift corrector for transmission mask
    tran_to_fluo_shift = ShiftCorrector()
    tran_to_fluo_shift.tvec = (-14, -12)
    army['Trans'].add_shift_corrector(tran_to_fluo_shift)

    # Add background correctors
    rb_fluo = RollingBallCorrector(600)
    army['Fluo'].add_background_corrector(rb_fluo)
    rb_auto = RollingBallCorrector(600)
    army['Auto'].add_background_corrector(rb_auto)
    rb_red = RollingBallCorrector(600)
    army['Red'].add_background_corrector(rb_red)

    # Add bleeding correction between Fluo and Auto
    auto_in_fluo_bleed = BleedingCorrector()
    auto_in_fluo_bleed.bleed_mean = 0.23
    auto_in_fluo_bleed.bleed_error = 0.06
    army['Fluo'].add_bleeding_corrector(auto_in_fluo_bleed, 'Auto')

    return army


def correct_stacks(trans, fluo, auto):
    army = create_army()

    army['Trans'].load_stack(trans.copy())
    army['Fluo'].load_stack(fluo.copy())
    army['Auto'].load_stack(auto.copy())
    army.run_correctors()
    return army['Trans'].stack, army['Fluo'].stack, army['Auto'].stack


def get_fluorescence_estimators(fluo_stack, mask):
    """Generates a dictionary with all the fluorescence descriptors.

    Parameters
    ----------
    fluo_stack : numpy.ndarray
        Array of the images with the intensity values
    mask : numpy.ndarray
        Array with masks for the images

    Returns
    -------
    estimators : dictionary
        Dictionary with values saved.
        'total_mean' : float
            mean inside the mask
        'border_mean' : float
            mean of the border of the mask
        'border_median' : float
            median of border of the mask
        'otsu_mean' : float
            mean of the otsu filtered border of the mask
        'otsu_median' : float
            median of the otsu filtered border of the mask
        'otsu_area' : int
            Area of the otsu filtered border of the mask
        'border_values' : numpy.ndarray
            Ordered array of the intensity values of the border of the mask
    """
    # Total mean
    mask = mask.astype(bool)
    estimators = {'total_mean': np.nanmean(fluo_stack[mask])}

    # Estimators of epithelia
    border_vals = get_border_vals(fluo_stack, mask)
    estimators['border_mean'] = np.nanmean(border_vals)
    estimators['border_median'] = np.nanmedian(border_vals)

    # Otsu filtered estimator
    otsu_vals = get_otsu_vals(fluo_stack, mask)
    estimators['otsu_mean'] = np.nanmean(otsu_vals)
    estimators['otsu_median'] = np.nanmedian(otsu_vals)
    estimators['otsu_area'] = len(otsu_vals.flatten())

    # Ordered border intensity values
    estimators['border_values'] = get_border_ordered_intensity(fluo_stack,
                                                               mask)

    return estimators


def get_border_vals(img, mask, thickness=10):
    """Returns list of values corresponding to the edge of the mask."""
    if img.ndim > 2:
        vals = np.asarray([get_border_vals(this_img, this_mask, thickness)
                           for this_img, this_mask in zip(img, mask)])
    else:
        new_mask = np.logical_xor(mask, morph.binary_erosion(mask,
                                                             selem=morph.disk(
                                                                 thickness)))
        vals = img[new_mask]

    return vals


def get_otsu_vals(img, mask, opening_radius=5):
    """Returns intensity values from img that are above Otsu thresholding img
    in mask."""
    thresh = filters.threshold_otsu(img[mask].flatten())
    new_mask = np.logical_and(mask, img > thresh)
    if img.ndim > 2:
        new_mask = np.asarray([morph.binary_opening(this,
                                                    selem=morph.disk(opening_radius))
                           for this in new_mask])
    else:
        morph.binary_opening(new_mask, selem=morph.disk(opening_radius))
    vals = img[new_mask]

    return vals


def get_border_ordered_intensity(img, mask=None, border_coords=None):
    """Order coordinates of border and returns an ordered list of values."""
    if img.ndim > 2:
        if mask is None:
            mask = [None] * len(img)
        if border_coords is None:
            border_coords = [None] * len(img)
        vals = np.asarray([get_border_ordered_intensity(this_img,
                                                        this_mask,
                                                        this_border_coords)
                           for this_img, this_mask, this_border_coords
                           in zip(img, mask, border_coords)])

    else:
        if mask is None and border_coords is None:
            raise ValueError('Need to provide mask or border coordinates')

        if mask is not None:
            border_coords = morpho.mask_to_snake(mask)

        sorted_coords = morpho.sort_border(border_coords)
        sorted_coords = tuple(map(tuple, sorted_coords.T[::-1]))
        vals = filters.gaussian(img, sigma=(10, 10))[sorted_coords]

    return vals
