import pathlib
import yaml
import multiprocessing

import numpy as np
import pandas as pd

from . import classifier as clf
from . import orgapath as op
from . import morpho
from . import image_manager as im
from . import visvis as vv


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
        self.max_images_buffered_time = 2  # Maximum number of images buffered
        # for time cropping
        self.max_images_buffered_region = 10  # Maximum number of images
        # buffered for region cropping
        self.workers = 5  # How many threads can be used

        # Set normalizers and classifiers
        self.normalizer = None
        self.classifier = None

    def set_output_path_and_load_df(self):
        """Looks for existing saved pandas files, loads them if existent and
        sets a savepath so as to not overwrite the previous file, unless
        overwrite attribute is True."""
        self.output_path = self.filepath_yaml.parent
        self.output_path = self.output_path.joinpath(self.output_name
                                                     + '.pandas')
        if self.output_path.exists():
            print('Loading previous DataFrame: %s' % str(self.output_path))
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
                                crop_filename=self.filepath_yaml_crop,
                                max_images_buffered_time=self.max_images_buffered_time,
                                max_images_buffered_region=self.max_images_buffered_region)
        else:
            op.add_crop_to_yaml(str(self.filepath_yaml),
                                max_images_buffered_time=self.max_images_buffered_time,
                                max_images_buffered_region=self.max_images_buffered_region)

        self.file_dict = self.load_yaml(self.filepath_yaml_crop)

    def analyze(self):
        """Analyzes every stack included in the file dictionary."""

        for file in self.file_dict.keys():
            fluo_file = self.file_dict[file]['yfp']
            auto_file = self.file_dict[file]['auto']
            region = self.file_dict[file]['crop']
            last_time = self.file_dict[file].get('time_crop')

            file, fluo_file, auto_file = self._check_path((file, fluo_file,
                                                           auto_file))

            print('Analyzing file: %s' % file)

            if self.df is not None and str(file) in self.df.tran_path.values:
                print('%s has already been analyzed' % file)
                continue

            this_file_res = self._analyze_file(file, fluo_file, auto_file,
                                               region, last_time)

            print('Saving file: %s' % file)

            this_df = pd.DataFrame(this_file_res)

            if self.df is not None:
                self.df = self.df.append(this_df,  ignore_index=True)
            else:
                self.df = this_df.copy()
            self.save_results()

    def _analyze_file(self, filepath, fluo_filepath, auto_filepath, region,
                      last_time):
        """Multiprocesses the analysis over a complete stack.

            Parameters
            ----------
            filepath : str
                path to the transmission stack
            fluo_filepath : str
                path to the fluorescence stack
            auto_filepath : str
                path to the fluorescence stack
            region : list, tuple
                coordinates of the cropped region
            last_time : int, None
                Length of stack to be considered. If None, the complete length
                is considered

            Returns
            -------
            df : pandas DataFrame
                DataFrame containing all the results from the analysis"""

        with multiprocessing.Pool(self.workers) as p:
            file_results = []
            for this_df in p.imap_unordered(morpho.timepoint_to_df,
                                            _my_iterator(filepath,
                                                         fluo_filepath,
                                                         auto_filepath,
                                                         region,
                                                         last_time)):
                file_results.append(this_df)

        df = pd.concat(file_results, ignore_index=True)
        df.sort_values('timepoint', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def update(self):
        """Deprecated method."""
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

    def describe_better(self):
        """Uses Haralick features to estimate the best plane in focus and
        estimates the Hu moments of this plane. columns added: haralick_n,
        z_best_haralick_n, focus_plane (boolean) and best_hu_moments (list)."""

        for num in range(1, 14):
            self.df['haralick_' + str(num) + '_best'] = np.nan
            self.df['z_best_haralick_' + str(num)] = np.nan
            self.df['focus_plane'] = False
        for j in range(1, 8):
            self.df['hu_moments_%s_best' % str(j)] = np.nan

        z_prev = 3
        z_hu = 0
        cols = ['moments_hu'] + ['haralick_' + str(n) for n in range(1, 14)]
        for params, this_df in self.df.groupby(['tran_path', 'timepoint']):
            print('For File %s polishing timepoint %s' % params)

            this_df.dropna(subset=cols, inplace=True)
            z = this_df.z.values

            # Best Haralick Features

            z_bests = []
            for num in range(1, 14):
                hara = this_df['haralick_' + str(num)].values
                z_best, hara_best = morpho.best_haralick(z, hara)

                z_bests.append(z_best)
                for i in this_df.index:
                    self.df.at[i, 'z_best_haralick_' + str(num)] = z_best
                    self.df.at[i, 'haralick_' + str(num) + '_best'] = hara_best

            # Best Focus Plane

            focus_plane = morpho.best_z_plane(z_bests,
                                              z_min=min(z), z_max=max(z),
                                              z_best_prev=z_prev)
            z_prev = focus_plane

            ind = this_df.index[this_df['z'] == focus_plane]
            # TODO: assert index exists
            self.df.at[ind, 'focus_plane'] = True

            # Hu Moments
            hu_matrix = np.stack(this_df.moments_hu.values)

            z_hu, hus = morpho.best_hu(z, hu_matrix, z_best_prev=z_hu)

            for i in this_df.index:
                for j in range(1, 8):
                    self.df.at[i, 'hu_moment_%s_best' % str(j)] = hus[j-1]

        self.estimate_total_fluorescence()

        self.save_results()

    def estimate_total_fluorescence(self):
        """Estimates Fluorescence by averaging Otsu mean values from different
        planes."""
        self.df['sum'] = self.df['otsu_mean'].values * self.df['otsu_area'].values

        self.df['temp_0'] = self.df.groupby(
            ['tran_path', 'timepoint'])['sum'].transform('sum')
        self.df['temp_1'] = self.df.groupby(
            ['tran_path', 'timepoint'])['otsu_area'].transform('sum')

        self.df['total_otsu_mean'] = self.df['temp_0'] / self.df['temp_1']

        self.df = self.df.drop(['temp_0', 'temp_1'], axis=1)

    def _check_path(self, paths):
        """Checks whether the given paths are in the same filepath as the yaml
        dictionary. If they're not, then the path is corrected.

        Parameters
        ----------
        paths : list, tuple, string, pathlib.Path
            Path to check if parent is shared

        Returns
        -------
        new_paths : list of paths or pathlib.Path
            Paths changed according to filepath of dictionary
        """
        if isinstance(paths, (list, tuple, np.ndarray)):
            new_paths = []
            for path in paths:
                new_path = self._check_path(path)
                new_paths.append(new_path)

        else:
            paths = pathlib.Path(paths)
            actual_parent = self.filepath_yaml_crop.parent
            path_parent = paths.parent
            if actual_parent != path_parent:
                new_paths = actual_parent.joinpath(paths.name)
            else:
                new_paths = paths

        return new_paths

    def load_normalizer(self, path):
        """Loads normalizer from path"""
        normalizer = clf.Normalizer()
        normalizer.load(path)
        self.normalizer = normalizer

    def load_classifier(self, path):
        """Loads classifier from path"""
        classifier = clf.Classifier()
        classifier.load(path)
        self.classifier = classifier

    def classify(self):
        """Runs the normalizer and classifier over every timepoint."""
        if self.normalizer is None:
            raise ValueError('No normalizer was loaded.')
        if self.classifier is None:
            raise ValueError('No classifier was loaded.')

        self.normalizer.normalize(self.df)
        self.df['state'] = self.classifier.classify(self.df)

        self.save_results()

    def plot_border_int_gif(self):
        """Generates a folder with all the gifs showing the intensity of border
        pixels."""

        save_path = self.output_path.parent.with_name(
            self.output_path.parent.name + '_border_intensities')

        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for this_file, this_df in self.df.groupby('tran_path'):
            fluo_file = this_df.fluo_path.values[0]
            auto_file = this_df.auto_path.values[0]
            paths = self._check_path((this_file, fluo_file, auto_file))

            this_df = this_df.query('focus_plane')

            save_dir = save_path.joinpath(paths[1].stem + '_border_int.gif')

            vv.make_border_int_gif(this_df, paths, save_dir)

    def plot_segmentation_and_state_gif(self):
        """Generates a folder with all the gifs showing transmission,
        fluorescence, segmentation, classification and fluorescence
        estimation."""

        save_path = self.output_path.parent.with_name(
            self.output_path.parent.name + '_segmentation_and_state')

        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for this_file, this_df in self.df.groupby('tran_path'):
            fluo_file = this_df.fluo_path.values[0]
            auto_file = this_df.auto_path.values[0]
            paths = self._check_path((this_file, fluo_file, auto_file))

            this_df = this_df.query('focus_plane')

            save_dir = save_path.joinpath(paths[1].stem + '_border_int.gif')

            vv.make_segmentation_and_state_gif(this_df, paths, save_dir)


def _my_iterator(filepath, fluo_filepath, auto_filepath, region, last_time):
    """Generates an iterator over the stack of images to use for
    multiprocessing."""

    keys = im.get_keys(filepath, last_time)

    for ndx, (key) in enumerate(keys):
        yield ndx, key, filepath, fluo_filepath, auto_filepath, region
