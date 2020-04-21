import pathlib
import queue
import time
from threading import Thread
import yaml

import numpy as np
from skimage import io
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool
from skimage.viewer.plugins.base import Plugin

from . import visvis as vv
from . import image_manager as im


def scan_folder(folder):
    """Iterator yielding every pair of transmission and yfp stacks in folder."""
    p = pathlib.Path(folder)
    for yfp in p.rglob('*YFP*'):
        if yfp.is_dir() or yfp.suffix.lower() != '.tif':
            continue
        trans = yfp.with_name(yfp.name.replace('YFP', 'Trans'))
        if not trans.exists():
            trans = None
        auto = yfp.with_name(yfp.name.replace('YFP', 'CFPYFP'))
        if not auto.exists():
            auto = None
        yield trans, yfp, auto


def create_base_yaml(folder, output):
    """Generates a yaml dictionary at output with all the pairs of stacks found
    at folder."""
    append_to_yaml(folder, output, {})


def append_to_yaml(folder, output, filename_or_dict):
    """Appends pairs of stacks found at folder to the yaml dictionary at
    filename_or_dict (or dictionary) and saves it
    at output.

    Parameters
    ----------
    folder : str
        path to folder where stacks are located
    output : str
        path to yaml file where dictionary is to be saved
    filename_or_dict : str or dict
        path to yaml dictionary or dictionary to which new stack paths are to be
        appended
    """

    if isinstance(filename_or_dict, str):
        with open(filename_or_dict, 'r', encoding='utf-8') as fi:
            d = yaml.load(fi.read())
    else:
        d = filename_or_dict

    for trans, yfp, auto in scan_folder(folder):
        d[str(trans)] = {'yfp': str(yfp), 'auto': str(auto)}

    with open(output, 'w', encoding='utf-8') as fo:
        fo.write(yaml.dump(d))


def test_yaml(filename):
    """Checks whether files at the dictionary exist in the saved path and if
    crop coordinates are saved."""
    with open(filename, 'r', encoding='utf-8') as fi:
        d = yaml.load(fi.read())
    cnt = 0
    cnt3 = 0
    for k, v in d.items():
        ok1, ok2 = pathlib.Path(k).exists(), pathlib.Path(v['yfp']).exists()
        ok3 = 'crop' in v
        print(k)
        print('- Trans: %s, YFP: %s, Crop: %s' % (ok1, ok2, ok3))
        cnt += 1 if ok1 and ok2 else 0
        cnt3 += 1 if ok3 else 0
    print('---')
    print('%d out of %d. Crop %d' % (cnt, len(d), cnt3))


# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class _background_consumer(Thread):
    """Will fill the queue with content of the source in a separate thread.

    >>> import Queue
    >>> q = Queue.Queue()
    >>> c = _background_consumer(q, range(3))
    >>> c.start()
    >>> q.get(True, 1)
    0
    >>> q.get(True, 1)
    1
    >>> q.get(True, 1)
    2
    >>> q.get(True, 1) is _sentinel
    True
    """

    def __init__(self, queue, source):
        Thread.__init__(self)

        self._queue = queue
        self._source = source

    def run(self):
        for item in self._source:
            self._queue.put(item)

        # Signal the consumer we are done.
        self._queue.put(_sentinel)


class ibuffer(object):
    """Buffers content of an iterator polling the contents of the given
    iterator in a separate thread.
    When the consumer is faster than many producers, this kind of
    concurrency and buffering makes sense.

    The size parameter is the number of elements to buffer.

    The source must be threadsafe.

    Next is a slow task:
    >>> from itertools import chain
    >>> import time
    >>> def slow_source():
    ...     for i in range(10):
    ...         time.sleep(0.1)
    ...         yield i
    ...
    >>>
    >>> t0 = time.time()
    >>> max(chain(*( slow_source() for _ in range(10) )))
    9
    >>> int(time.time() - t0)
    10

    Now with the ibuffer:
    >>> t0 = time.time()
    >>> max(chain(*( ibuffer(5, slow_source()) for _ in range(10) )))
    9
    >>> int(time.time() - t0)
    4

    60% FASTER!!!!!11
    """

    def __init__(self, size, source):
        self._queue = queue.Queue(size)

        self._poller = _background_consumer(self._queue, source)
        self._poller.daemon = True
        self._poller.start()

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get(True)
        if item is _sentinel:
            raise StopIteration()
        return item


class Timer(object):
    """As clear as the name."""

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.perf_counter()


def load_mp_image(filenames, dcrop=None):
    """Iterates over the given list of filenames and yields filename and either
    the saved crop coordinates (if dcrop has them), a normalized image to
    perform the crop or None if 'single clones' is inside filename or errors
    arise while getting the image."""
    for filename in filenames:
        k = filename
        if dcrop and k in dcrop and 'crop' in dcrop[k]:
            yield filename, dcrop[k]['crop']
        elif 'single clones' in filename:
            yield filename, None
        else:
            try:
                with Timer() as t:
                    if 'time_crop' in dcrop[k]:
                        original = im.load_stack(filename,
                                                 last_time=dcrop[k]['time_crop']
                                                 )
                    else:
                        original = io.imread(filename)
                    image = original.reshape(-1, *original.shape[-2:])
                    image = np.min(image, axis=0)
                    image = (image - np.min(image)) / \
                            (np.max(image) - np.min(image))
                print('%.2f: %s' % (t.elapsed, filename))
                yield filename, image
            except:
                print('time projection for rectangle selection failed')
                yield filename, None


def load_stack(filenames, dcrop=None):
    """Iterates over the given list of filenames and yields filename and either
    the saved crop coordinates (if dcrop has them), a normalized image to
    perform the crop or None if errors arise while getting the image."""
    for filename in filenames:
        k = filename
        if dcrop and k in dcrop and 'time_crop' in dcrop[k]:
            yield filename, dcrop[k]['time_crop']
        else:
            try:
                with Timer() as t:
                    original = io.imread(filename)

                print('%.2f: %s' % (t.elapsed, filename))
                yield filename, original
            except:
                print('load stack did not work')
                yield filename, None


def add_crop_to_yaml(filename, crop_filename=None, max_images_buffered_time=2,
                     max_images_buffered_region=10):
    """Opens filename dictionary and asks for a crop to be saved at filename +
    _crop. If crop_filename is given, then it checks whether a crop has been
    saved."""
    if crop_filename is not None:
        with open(crop_filename, 'r', encoding='utf-8') as fi:
            dcrop = yaml.load(fi.read())
    else:
        dcrop = None

    with open(filename, 'r', encoding='utf-8') as fi:
        dinput = yaml.load(fi.read())

    dout = {}

    # Manage Time crops
    try:
        for ndx, (k, image_or_crop) in \
                enumerate(ibuffer(max_images_buffered_time,
                                  load_stack(dinput.keys(), dcrop))):
            print('%d/%d: %s' % (ndx, len(dinput), k))
            v = dinput[k]

            if image_or_crop is None:
                pass

            elif isinstance(image_or_crop, int):
                v['time_crop'] = image_or_crop

            else:
                image = image_or_crop
                chosen_t = vv.visualizer(image)
                print(chosen_t)

                v['time_crop'] = chosen_t

            dout[k] = v

    except KeyboardInterrupt:
        pass

    # Manage Rectangle crops
    try:
        for ndx, (k, image_or_crop) in \
                enumerate(ibuffer(max_images_buffered_region, load_mp_image(dinput.keys(), dout))):
            print('%d/%d: %s' % (ndx, len(dinput), k))
            v = dinput[k]

            if image_or_crop is None:
                pass

            elif isinstance(image_or_crop, (list, tuple)):
                v['crop'] = image_or_crop

            else:
                image = image_or_crop
                viewer = ImageViewer(image)
                rect_tool = RectangleTool(viewer)
                viewer.show()
                x = list(float(f) for f in rect_tool.extents)

                print(image.shape, x)

                if x[0] == 0:
                    break

                v['crop'] = x

            dout[k] = v

    except KeyboardInterrupt:
        pass

    with open(filename[:filename.rfind('.')] + '_crop.yaml',
              'w', encoding='utf-8') as fo:
        fo.write(yaml.dump(dout))
