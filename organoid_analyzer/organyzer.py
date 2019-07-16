import pathlib
import yaml
import multiprocessing
import pandas as pd
import numpy as np

from organoid_analyzer import orgapath as op
from organoid_analyzer import morpho
from organoid_analyzer import image_manager as im

class Organyzer(object):

    def __init__(self, filepath_or_folder, output_name, overwrite=False):
        filepath = pathlib.Path(filepath_or_folder)
        self.output_name = output_name
        self.overwrite = overwrite  # True if overwriting pandas file is ok

        # set filepaths and load dictionary
        self.filepath_yaml = None
        self.filepath_yaml_crop = None
        self.file_dict = None
        self.set_filepaths_and_file_dicts(filepath)
        self.set_output_path_and_load_df()

        # set parameters for analysis and saving files
        self.workers = 5  # How many threads can be used

    def set_output_path_and_load_df(self):
        """Looks for existing saved pandas files, loads them if existent and
        sets a savepath so as to not overwrite the previous file, unless
        overwrite attribute is True."""
        self.output_path = self.filepath_yaml.parent
        self.output_path = self.output_path.joinpath(self.output_name
                                                     + '.pandas')
        if self.output_path.exists():
            self.df = self.load_pandas()

            if self.overwrite:
                return

            if not self.output_path.stem.split('_')[-1].isdigit():
                self.output_path = self.output_path.with_name(
                    self.output_name + '_0.pandas')

            file_num = int(self.output_path.name.split('_')[-1].split('.')[0])
            while self.output_path.exists():
                self.df = self.load_pandas()
                self.output_path = self.output_path.with_name(
                    self.output_name + '_' + str(file_num) + '.pandas')
                file_num += 1

        else:
            self.df = None

    def load_pandas(self):
        """Load saved pandas file"""
        return pd.read_pickle(str(self.output_path))

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
            if self.filepath_yaml.exists():
                self.file_dict = self.load_yaml(self.filepath_yaml)

                op.append_to_yaml(str(filepath), self.filepath_yaml,
                                  self.file_dict)

            else:
                op.create_base_yaml(str(filepath), str(self.filepath_yaml))

            self.filepath_yaml_crop = \
                self.filepath_yaml.with_name(self.filepath_yaml.stem +
                                             '_crop.yaml')

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
            self.filepath_yaml_crop = \
                self.filepath_yaml.with_name(self.filepath_yaml.stem +
                                             '_crop.yaml')

        if self.filepath_yaml_crop and self.filepath_yaml_crop.exists():
            self.file_dict = self.load_yaml(self.filepath_yaml_crop)
        else:
            self.file_dict = self.load_yaml(self.filepath_yaml)

    def load_yaml(self, path):
        """Loads a yaml file into a dictionary"""
        with open(path, 'r', encoding='utf-8') as fi:
            file_dict = yaml.load(fi.read())
        return file_dict

    def save_results(self):
        """Save DataFrame containing results."""
        self.df.to_pickle(str(self.output_path))

    def crop(self):
        """Asks for the cropping of the listed files, saves the crop yaml and
        loads the dictionary with the crops."""
        if self.filepath_yaml_crop.exists():
            op.add_crop_to_yaml(str(self.filepath_yaml),
                                crop_filename=self.filepath_yaml_crop)
        else:
            op.add_crop_to_yaml(str(self.filepath_yaml))

        self.file_dict = self.load_yaml(self.filepath_yaml_crop)

    def analyze(self):
        """Analyzes every stack included in the file dictionary."""

        for file in self.file_dict.keys():
            fluo_file = self.file_dict[file]['yfp']
            region = self.file_dict[file]['crop']

            print('Analyzing file: %s' % file)

            if self.df is not None and file in self.df.tran_path.values:
                print('%s has already been analyzed' % file)
                continue

            this_file_res = self._analyze_file(file, fluo_file, region)

            print('Saving file: %s' % file)

            this_df = pd.DataFrame(this_file_res)

            if self.df is not None:
                self.df = self.df.append(this_df,  ignore_index=True)
            else:
                self.df = this_df.copy()
            self.save_results()

    def _analyze_file(self, filepath, fluo_filepath, region):
        """Multiprocesses the analysis over a complete stack.

            Parameters
            ----------
            filepath : str
                path to the transmission stack
            fluo_filepath : str
                path to the fluorescence stack
            region : list, tuple
                coordinates of the cropped region

            Returns
            -------
            df : pandas DataFrame
                DataFrame containing all the results from the analysis"""

        with multiprocessing.Pool(self.workers) as p:
            file_results = []
            for this_df in p.imap_unordered(morpho.timepoint_to_df,
                                            _my_iterator(filepath,
                                                         fluo_filepath,
                                                         region)):
                file_results.append(this_df)

        df = pd.concat(file_results, ignore_index=True)
        df.sort_values('timepoint', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def update(self):
        this_file_res = []

        for file in self.file_dict.keys():
            print('Analyzing file: %s' % file)

            sel_df = self.df.query('tran_path == "%s"' % file)
            tran_stack = morpho.skio.imread(file)
            index = sel_df.timepoint.values
            snakes = sel_df.external_snakes.values

            for (ndx, tran0, snake) in zip(index, tran_stack, snakes):

                df = pd.DataFrame({'tran_path': file, 'timepoint': [ndx]})
                description = morpho.get_texture_description(tran0, snake)

                for prop in description.keys():
                    if isinstance(description[prop], (tuple, list, np.ndarray)):
                        df[prop] = [description[prop]]
                    else:
                        df[prop] = description[prop]

                this_file_res.append(df)

            print('Merging file: %s' % file)

        all_df = pd.concat(this_file_res, ignore_index=True)

        if self.df is not None:
            self.df = self.df.merge(all_df,  on=['tran_path', 'timepoint'])
        else:
            self.df = all_df.copy()
        self.save_results()


def _my_iterator(filepath, fluo_filepath, region):
    """Generates an iterator over the stack of images to use for
    multiprocessing."""

    tran_meta = im.get_metadata(filepath)

    try:
        times = int(tran_meta['time'])
        z = int(tran_meta['z'])
    except KeyError:
        times = int(tran_meta['frames'])
        z = int(tran_meta['slices'])

    keys = np.arange(times * z).reshape(times, z)

    for ndx, (key) in enumerate(keys):
        yield ndx, key, filepath, fluo_filepath, region
