import pathlib
import yaml
import multiprocessing
import pandas as pd
import numpy as np

from organoid_analyzer import orgapath as op
from organoid_analyzer import morpho


class Organyzer(object):

    def __init__(self, filepath_or_folder, output_name):
        filepath = pathlib.Path(filepath_or_folder)
        self.output_name = output_name
        self.output_path = filepath.parent
        self.output_path = self.output_path.joinpath(self.output_name
                                                     + '.pandas')
        # set filepaths and load dictionary
        self.filepath_yaml = None
        self.filepath_yaml_crop = None
        self.file_dict = None
        self.set_filepaths_and_file_dicts(filepath)

        # set parameters for analysis and saving files
        self.overwrite = False  # True if overwriting pandas file is ok
        self.workers = 5  # How many threads can be used
        self.df = None

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

    def save_results(self):
        """Save DataFrame containing results. Checks before overwriting."""
        if self.overwrite:
            save_path = str(self.output_path)
        else:
            save_path = self.output_path
            file_num = 0
            while save_path.exists():
                save_path = save_path.with_name(save_path.stem + '_' +
                                                str(file_num) + '.pandas')
                file_num += 1
            save_path = str(save_path)

        self.df.to_pickle(save_path)

    def crop(self):
        """Asks for the cropping of the listed files, saves the crop yaml and
        loads the dictionary with the crops."""
        op.add_crop_to_yaml(str(self.filepath_yaml),
                            crop_filename=self.filepath_yaml_crop)
        self.filepath_yaml_crop = \
            self.filepath_yaml.with_name(self.filepath_yaml.stem +
                                         '_crop.yaml')
        self.file_dict = self.load_yaml(self.filepath_yaml_crop)

    def analyze(self):
        """Analyzes every stack included in the file dictionary."""
        all_dfs = []
        for file in self.file_dict.keys():
            # TODO: Check already analyzed files
            fluo_file = self.file_dict[file]['yfp']
            region = self.file_dict[file]['crop']

            print('Analyzing file: %s' % file)

            this_file_res = self._analyze_file(file, fluo_file, region)

            print('Saving file: %s' % file)

            all_dfs.append(this_file_res)

        self.df = pd.concat(all_dfs, ignore_index=True)
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

        tran_stack = morpho.skio.imread(filepath)
        fluo_stack = morpho.skio.imread(fluo_filepath)

        print('I am going to analyze %s timepoints.' % (len(tran_stack)))

        with multiprocessing.Pool(self.workers) as p:
            file_results = []
            for this_df in p.imap_unordered(self.timepoint_to_df,
                                            self._my_iterator(tran_stack,
                                                              fluo_stack,
                                                              filepath,
                                                              fluo_filepath,
                                                              region)):
                file_results.append(this_df)

        df = pd.concat(file_results, ignore_index=True)
        df.sort_values('timepoint', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def _my_iterator(self, tran_stack, fluo_stack,
                     filepath, fluo_filepath, region):
        """Generates an iterator over the stack of images to use for
        multiprocessing."""
        for ndx, (tran0, fluo0) in enumerate(zip(tran_stack[:2], fluo_stack)):
            yield ndx, tran0, fluo0, filepath, fluo_filepath, region

    def timepoint_to_df(self, params):
        """Analyzes a single timepoint and generates a small pandas DataFrame.

        Parameters
        ----------
        params : list
            ndx, tran, fluo, region, filepath, fluo_filepath

        Returns
        -------
        df : pandas DataFrame
            Small DataFrame with the results of a single timepoint analysis."""

        ndx, tran, fluo, filepath, fluo_filepath, region = params

        print('analyzing timepoint %s from file %s' % (ndx, filepath))

        to_save = morpho.segment_timepoint(tran, fluo, region)
        mask = morpho.snake_to_mask(to_save['external_snakes'][0], tran.shape)
        description = morpho.get_description(mask)

        df = pd.DataFrame(to_save)

        for prop in description.keys():
            if isinstance(description[prop], (tuple, list, np.ndarray)):
                df[prop] = [description[prop]]
            else:
                df[prop] = description[prop]

        df['tran_path'] = filepath
        df['fluo_path'] = fluo_filepath
        df['crop'] = [region]
        df['timepoint'] = ndx

        return df
