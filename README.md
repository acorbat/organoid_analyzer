# Organyzer

**(Class) Analyzer(filepath_or_folder, output_name):** On creation, it receives either a filepath to the yaml file where a dictionary of all the files that are to be analyzed is created or folder where the stacks are and creates and saves the dictionary.

## Usage
*(method) crop:* Shows not cropped stacks from the yaml file dictionary and lets you add a crop to them.

*(method) analyze:* Analyzes the whole list of files inside the dictionary that have been cropped and saves the results into a pandas DataFrame.

The following script will load the file dictionary (or create it if missing), show the averaged timelapsed images to crop (if crops are missing) and analyze every ile and save it to a pandas DataFrme with the output name.

    from organoid_analyzer import organyzer
    org = organyzer.Organyzer('path/to/folder', 'name_to_save')
    org.crop()
    org.analyze()
   
   
# ImageOrganyzer

**(Class) ImageOrganyzer(folder):** On creation, it receives a filepath to the folder containing subfolders that share the same name but ending with the number corresponding to how they were acquired (i.e. date_001). If there is a "regions" folder it is used to create PA images and a csv file.

## Usage
*(method) concatenate_and_reshape:* Gets the ordered list of folders, loads the images inside them and concatenates and reshapes them according to metadata. These are saved in a folder at the same location with the "_concatenated" appended.

*(method) plot_and_save_photoactivation:* Looks into the regions folder at location and plots the selected photoactivation region, as well as saving a csv file with the selected regions.

The following script will generate the concatenated files from the subfolders in path/to/folder and generate the photoactivation images and csv file if a regions folder exists.

    from organoid_analyzer import image_manager as man
    
    manager = man.ImageOrganyzer('path/to/folder')

    manager.concatenate_and_reshape()

    if manager.regions_paths.exists():
        manager.plot_and_save_photoactivation()


# Orgapath

Orgapath is the library used to generate the yaml dictionaries where locations and coordinates of crops are to be saved.

## Usage

**create_base_yaml(folder, output):** searches for pairs of stacks named Trans and YFP and generates a yaml file where locations of each pair is saved in the output path.

**add_crop_to_yaml(filename, crop_filename=None):** reads the generated yaml file and looks for already performed crops at the crop_filename path. If there are missing crop coordinates or crop_filename is not given, then images are shown in order to manually generate the crops for them.


# Morpho

Morpho is the library containing the segmentation and morphometrics functions. 

## Usage

**segment_timepoint(tran, fluo, region)** receives a transmission single image, fluorescence single image and a list of coordinates for the cropping region. It applies a sequence of segmentation steps to find contours of the files. A dictionary for each contour this single frame is returned.

This function is called for each timepoint in the stack loaded by the Organyzer.

**get_description(mask, descriptors=None):** It returns a dictionary with the list of descriptors calculated from the mask. If no lsit of descriptors is chosen then the default list is ['area', 'centroid', 'convex_area', 'eccentricity',
                       'equivalent_diameter', 'euler_number', 'extent',
                       'major_axis_length', 'minor_axis_length', 'moments_hu',
                       'perimeter', 'solidity'].

This function is called for each timepoint in the stack loaded by the Organyzer.
