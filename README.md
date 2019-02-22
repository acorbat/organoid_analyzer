# Orgapath

Orgapath is the library used to generate the yaml dictionaries where locations and coordinates of crops are to be saved.

## Usage

__create_base_yaml(folder, output):__ searches for pairs of stacks named Trans and YFP and generates a yaml file where locations of each pair is saved in the output path.

__add_crop_to_yaml(filename, crop_filename=None):__ reads the generated yaml file and looks for already performed crops at the crop_filename path. If there are missing crop coordinates or crop_filename is not given, then images are shown in order to manually generate the crops for them.

