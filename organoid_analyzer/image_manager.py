import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import draw, transform, util, filters

import tifffile as tif


class ImageOrganyzer(object):

    def __init__(self, folder):
        self.overwrite = False
        self.path = pathlib.Path(folder)
        self.folder_name = self.path.stem
        self.save_path = self.path.parent.joinpath(self.folder_name +
                                                   '_concatenated')
        self.inner_folders = self.get_folder_list()

        self.regions_paths = self.path.joinpath('regions')

    def get_folder_list(self):
        """Returns an ordered list of all the folders inside the main experiment
         folder"""
        folders = [folder for folder in self.path.iterdir()
                   if self.folder_name in folder.stem]
        folders = {int(this.stem.split('_')[-1]): this for this in folders}
        folder_numbers = list(folders.keys())
        folder_numbers = sorted(folder_numbers)
        folders = [folders[ind] for ind in folder_numbers]

        print("I will concatenate files found in the following folders:")
        for folder in folders:
            print(folder)

        return folders

    def save_img(self, save_path, stack, axes='YX', create_dir=False,
                 metadata=None):
        """Saves stack as 16-bit integer in tif format."""
        stack = stack.astype('int16')

        # Fill array with new axis
        ndims = len(stack.shape)
        while ndims < 5:
            stack = stack[np.newaxis, :]
            ndims = len(stack.shape)

        # Add missing and correct axes order according to fiji
        new_axes = [ax for ax in 'TZXCY' if ax not in axes[:]]
        axes = ''.join(new_axes) + axes

        stack = tif.transpose_axes(stack, axes, asaxes='TZCYX')

        if create_dir and not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        tif.imsave(str(save_path), data=stack, imagej=True, metadata=metadata)

    def concatenate_and_reshape(self):
        first_folder = self.path.joinpath(self.inner_folders[0])
        for this_file in first_folder.iterdir():
            print('Concatenating %s' % str(this_file))
            if this_file.suffix.lower() != '.tif' and \
                    this_file.suffix.lower() != '.btf':
                continue

            this_file_name = this_file.name
            save_path = self.save_path.joinpath(this_file_name)
            save_path = save_path.with_suffix('.tif')
            if save_path.exists():
                print('File already concatenated')

            stacks = []
            metadatas = []
            times = []
            zetas = []

            for folder in self.inner_folders:
                this_other_file = folder.joinpath(this_file_name)

                print('with ' + str(this_other_file))

                if not this_other_file.exists():
                    print("file not found")
                    continue

                metadata = get_metadata(str(this_other_file))
                metadatas.append(metadata)
                this_img_file = tif.TiffFile(str(this_other_file))

                times.append(int(metadata['time']))
                zetas.append(int(metadata['z']))

                stack = this_img_file.asarray()

                stacks.append(stack)

            stack = np.concatenate(stacks)
            times = np.sum(times)
            assert all(x == zetas[0] for x in zetas)
            zetas = zetas[0]

            stack = stack.reshape(times, zetas, stack.shape[-2], stack.shape[-1])
            metadata = metadatas[-1]
            metadata['time'] = times

            self.save_img(save_path, stack, axes='TZYX', create_dir=True,
            metadata=metadata)

    def get_region(self, filepath):
        with open(str(filepath), 'rb') as file:
            for row in file:
                this_r = row.decode("utf-8")
                if 'Region Points' in this_r:
                    region_text = this_r.split('=')[-1]
                    region_text = region_text.split(';')[:-1]
                    bbox = [this.split(',') for this in region_text]
                    new_bbox = [0, ] * 4
                    for n, dim in enumerate(bbox):
                        new_bbox[n:n+3:2] = [int(elem) for elem in dim[::-1]]
                    return new_bbox

    def get_photoactivation_mask(self, filepath, img):
        bbox = self.get_region(str(filepath))
        coords = draw.rectangle(bbox[0:3:2], end=bbox[1:4:2], shape=img.shape)
        mask = np.zeros(img.shape)
        mask[coords] = 1
        return mask

    def plot_and_save_photoactivation(self):
        save_path = self.path.parent.joinpath(self.folder_name + '_regions')
        save_path.mkdir(exist_ok=True)
        this_dict = {'filename': [], 'region': []}
        for file in self.regions_paths.iterdir():
            if file.suffix != '.rgn':
                continue

            img_file = tif.TiffFile(str(file.with_suffix('.tif')))
            img = img_file.asarray()

            region = self.get_region(str(file))
            this_dict['filename'].append(file.stem)
            this_dict['region'].append(region)
            pa_mask = self.get_photoactivation_mask(str(file), img)

            plt.imshow(img, cmap='Greys')
            plt.contour(pa_mask, colors='r')
            plt.axis('off')

            this_img_save_path = save_path.joinpath(file.name)
            this_img_save_path = this_img_save_path.with_suffix('.png')
            plt.savefig(str(this_img_save_path))
            plt.close()

        df_regions = pd.DataFrame(this_dict)
        df_regions.to_csv(str(save_path.joinpath('regions.csv')))


def get_metadata(filepath):
    img = tif.TiffFile(str(filepath))
    if img.is_imagej:
        meta_dict = img.imagej_metadata

    else:
        with open(str(filepath), 'rb') as file:
            metadata = []
            append_it = False
            for row in file:
                try:
                    this_r = row.decode("utf-8")

                    if "Band" in this_r:
                        append_it = True
                    if append_it and "=" in this_r:
                        metadata.append(this_r)
                    if 'TimePos' in this_r:
                        append_it = False

                except:
                    pass
        meta_dict = {val.split('=')[0].lower(): val.split('=')[1].replace('\n', '') for
                     val in metadata}
        for key, val in meta_dict.items():
            try:
                meta_dict[key] = float(val)
            except:
                pass

    return meta_dict


def get_keys(filepath, last_time=None):
    tran_meta = get_metadata(filepath)

    try:
        times = int(tran_meta['time'])
        z = int(tran_meta['z'])
    except KeyError:
        times = int(tran_meta['frames'])
        z = int(tran_meta['slices'])

    if last_time:
        times = last_time + 1

    return np.arange(times * z).reshape(times, z)


def load_stack(filepath, last_time=None):
    keys = get_keys(filepath, last_time=last_time)
    stack = tif.TiffFile(str(filepath)).asarray(key=keys.flatten())
    stack = stack.reshape(*keys.shape, *stack.shape[-2:])

    return stack
