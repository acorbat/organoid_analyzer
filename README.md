# Organ(oid_anal)yzer

**(Class) Organyzer(filepath_or_folder, output_name):** On creation, it receives either a filepath to the yaml file where a dictionary of all the files that are to be analyzed is created or folder where the stacks are and creates and saves the dictionary.

## Usage
*(method) crop:* Shows not cropped stacks from the yaml file dictionary and lets you add a crop to them and select the las timepoint to be analyzed.

*(method) analyze:* Analyzes the whole list of files inside the dictionary that have been cropped and saves the results into a pandas DataFrame.

*(method) describe_better:* Based on previous segmentation, Hu moments and Haralick features looks for the best plane in focus and estimates the best descriptors for the organoids.

*(method) classify:* Applies a loaded normalizer and classifier to the organoid descriptors in order to generate a classification for each organoid.

*(method) plot_segmentation_and_state_gif:* Creates a folder in save_path ending with '_segmentation_and_state' where movies showing the segmentation and classification are saved.

*(method) plot_border_int_gif:* Creates a folder in save_path ending with '_border_int' where movies showing the segmentation and border intensities are saved.

The following script will load the file dictionary (or create it if missing), show the averaged timelapsed images to crop (if crops are missing) and lets you choose the last timepoint to be considered. This is later saved in a yaml dictionary that will be accessed by Organyzer in following steps. 
    
    from organoid_analyzer import organyzer
    org = organyzer.Organyzer('path/to/folder', 'name_to_save')
    org.crop()
   
Once all files have been cropped and timepoints selected, this script will analyze every file and save it to a pandas DataFrame with the output name. Consider running this step in a cluster as it is computationally expensive.

    from organoid_analyzer import organyzer
    org = organyzer.Organyzer('path/to/folder', 'name_to_save')
    org.analyze()
   
After generating the segmentation and some descriptors in the previous analysis, the next script should be run to improve description on analysis. This step will look for the best plane in focus, save the best Hu moments and Haralick features, as well as fluorescence estimations.

    from organoid_analyzer import organyzer
    org = organyzer.Organyzer('path/to/folder', 'name_to_save')
    org.describe_better()
    
For classification of organoids, we need to load a normalizer and a classifier and then run the corresponding code.

    from organoid_analyzer import organyzer
    org = organyzer.Organyzer('path/to/folder', 'name_to_save')
    org.load_normalizer('path_to_normalizer')
    org.load_classifier('path_to_classifier')
    org.classify()
    
Finally, with all the analysis and classification done, we can generate movies from the results by using the following script.

    from organoid_analyzer import organyzer
    org = organyzer.Organyzer('path/to/folder', 'name_to_save')
    org.plot_segmentation_and_state_gif()
    org.plot_border_int_gif()
    
This will automatically generate new folders ending in '_segmentation_and_state' and '_border_intensities' with the corresponding movies.

# ImageOrganyzer

**(Class) ImageOrganyzer(folder):** On creation, it receives a filepath to the folder containing subfolders that share the same name but ending with the number corresponding to how they were acquired (i.e. date_001). If there is a "regions" folder it is used to create PA images and a csv file.

## Usage
*(method) concatenate_and_reshape:* Gets the ordered list of folders, loads the images inside them and concatenates and reshapes them according to metadata. These are saved in a folder at the same location with "_concatenated" appended.

*(method) plot_and_save_photoactivation:* Looks into the regions folder at location and plots the selected photoactivation region, as well as saving a csv file with the selected regions.

The following script will generate the concatenated files from the subfolders in path/to/folder and generate the photoactivation images and csv file if a regions folder exists.

    from organoid_analyzer import image_manager as man
    
    manager = man.ImageOrganyzer('path/to/folder')

    manager.concatenate_and_reshape()

    if manager.regions_paths.exists():
        manager.plot_and_save_photoactivation()

### Notes

This package contains useful functions to get metadata and specific images by means of keys (such as z-plane and time).


# Orgapath

Orgapath is the library used to generate the yaml dictionaries where locations and coordinates of crops are to be saved.

## Usage

**create_base_yaml(folder, output):** searches for pairs of stacks named Trans and YFP and generates a yaml file where locations of each pair is saved in the output path.

**add_crop_to_yaml(filename, crop_filename=None):** reads the generated yaml file and looks for already performed crops at the crop_filename path. If there are missing crop coordinates or crop_filename is not given, then images are shown in order to manually generate the crops for them.


# Morpho

Morpho is the library containing the segmentation and morphometrics functions. 

## Usage

**timepoint_to_df(params):** is called by Organyzer for each timepoint and returns a small DataFrame with the results of a single timepoint analysis.

**segment_timepoint(tran, region):** receives a transmission single image and a list of coordinates for the cropping region. It applies a sequence of segmentation steps to find contours of the files. A dictionary for each contour this single frame is returned.

**generate_description(mask, trans, fluo, auto):** receives the mask, transmission image, fluorescence image and autofluorescence image and returns a dictionary with shape, texture and fluorescent descriptors.

**get_description(mask, descriptors=None):** It returns a dictionary with the list of descriptors calculated from the mask. If no lsit of descriptors is chosen then the default list is ['area', 'centroid', 'convex_area', 'eccentricity',
                       'equivalent_diameter', 'euler_number', 'extent',
                       'major_axis_length', 'minor_axis_length', 'moments_hu',
                       'perimeter', 'solidity', ].

This function is called for each timepoint in the stack loaded by the Organyzer.

# Classifier

**(Class) Normalizer():** Creates a normalizer object.

## Usage
*(method) find_normalization(df, col):* receives a DataFrame (df) and the list of columns (cols) to be normalized.

*(method) save(path):* Path to where the normalizer is to be saved.

*(method) load(path):* Path to where the normalizer is to be loaded from.


**(Class) Classifier():** Creates a classifier object.

## Usage
*(method) train(df):* receives a DataFrame (df) previously normalized trains the classifier.

*(method) classify(df):* returns a list of the classes predicted by the classifier.

*(method) save(path):* Path to where the normalizer is to be saved.

*(method) load(path):* Path to where the normalizer is to be loaded from.