import pathlib
import yaml

from organoid_analyzer import orgapath as op


class Analyzer(object):

    def __init__(self, filepath_or_folder, output_name):
        filepath = pathlib.Path(filepath_or_folder)
        self.output_name = output_name
        self.filepath_yaml = None
        self.filepath_yaml_crop = None
        self.file_dict = None

        self.set_filepaths_and_file_dicts(filepath)

    def set_filepaths_and_file_dicts(self, filepath):
        """Checks whether the filepath is a folder, base yaml or cropped yaml.
        If it a folder it creates a base yaml with the found stacks in the
        folder. If it is a yaml file, it looks for the word crop and sets the
        filepath. After that it loads the dictionary corresponding to the
        files."""

        if filepath.is_dir():
            print('I am going to look for stacks in this folder: '
                  + str(filepath))

            self.filepath_yaml = filepath.joinpath(self.output_name + '.yaml')
            op.create_base_yaml(str(filepath), str(self.filepath_yaml))

        elif 'crop' in filepath.stem:
            print('I am going to analyze already cropped stacks from this '
                  'file: ' + str(filepath))

            self.filepath_yaml_crop = pathlib.Path(filepath)
            self.filepath_yaml = self.filepath_yaml_crop.with_name(
                self.filepath_yaml_crop.name.replace("_crop", ""))
            if not self.filepath_yaml.exists():
                print('I did not find original not cropped yaml next to '
                      'cropped one and you will not be able to add crops')

        else:
            print('I am going to analyze stacks after you crop them from this '
                  'file: ' + str(filepath))

            self.filepath_yaml = pathlib.Path(filepath)

        if self.filepath_yaml_crop and self.filepath_yaml_crop.exists():
            self.file_dict = self.load_yaml(self.filepath_yaml_crop)
        else:
            self.file_dict = self.load_yaml(self.filepath_yaml)

    def load_yaml(self, path):
        """Loads a yaml file into a dictionary"""
        with open(path, 'r', encoding='utf-8') as fi:
            file_dict = yaml.load(fi.read())
        return file_dict

    def crop(self):
        """Asks for the cropping of the listed files, saves the crop yaml and
        loads the dictionary with the crops."""
        op.add_crop_to_yaml(str(self.filepath_yaml),
                            crop_filename=self.filepath_yaml_crop)
        self.filepath_yaml_crop = \
            self.filepath_yaml.with_name(self.filepath_yaml.stem +
                                         '_crop.yaml')
        self.file_dict = self.load_yaml(self.filepath_yaml_crop)
