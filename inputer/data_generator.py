from scipy.misc import imread
from utils.utils import load_formulas, get_logger
from utils.linux_wrapper import create_dir
from utils.image_operation import build_images


class DataGeneratorFile:
    """Simple Generator of tuples (img_path, formula_id)"""

    def __init__(self, filename):
        """Inits Data Generator File Generator

        Args:
            filename(str): path to file
        """
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as f:
            for line in f:
                line = line.strip().split(' ')
                path_img, id_formula = line[0], line[1]
                yield path_img, id_formula


class DataGenerator:
    """Data Generator of tuple (image, formula)

    Attributes:
        path_formulas（str): path to file of formulas
        dir_images(str): directory of images
        path_matching(str): file of matching pairs of (image filename, formula id)
        img_pre(lambda): image preprocess
        formula_pre(lambda): formulas preprocess
        max_iter(int): maximum numbers of elements in the dataset
        max_len(int): maximum length of a formula in the dataset
        iter_mode(str): 'data' or 'full'
        bucket(bool): decides if bucket the data by size of image
        bucket_size(int):

    """

    def __init__(self, path_formulas, dir_images, path_matching, bucket=False,
                 formula_pre=lambda s: s.strip().split(' '), iter_mode='data',
                 img_pre=lambda x: x, max_iter=None, max_len=None, bucket_size=20,
                 ):
        self._path_formulas = path_formulas
        self._dir_images = dir_images
        self._path_matching = path_matching
        self._img_pre = img_pre
        self._formula_pre = formula_pre
        self._max_iter = max_iter
        self._max_len = max_len
        self._iter_mode = iter_mode
        self._bucket = bucket
        self._bucket_size = bucket_size
        self._length = None
        self._formulas = self._load_formulas(path_formulas)
        self._set_data_generator()

    def _set_data_generator(self):
        """Sets iterable or generator of tuples (image path, formula id)"""
        self._data_generator = DataGeneratorFile(self._path_matching)

        if self._bucket:
            self._data_generator = self.bucket(self._bucket_size)

    def _load_formulas(self, filename):
        """Loads txt file with formulas in a dict

        Args:
            filename(str): path of formulas

        Returns:
            formulas(dict): id -> raw_formula
        """
        formulas = load_formulas(filename)
        return formulas

    def _get_raw_formula(self, formula_id):
        try:
            formula_raw = self._formulas[int(formula_id)]
        except KeyError:
            print("Tried to access id {} but only {} formulas".format(
                formula_id, len(self._formulas)))
            print("Possible fix: mismatch between matching file and formulas")
            raise KeyError

        return formula_raw

    def _process_instance(self, sample):
        """From path and formula id, returns actual data

        Applies preprocess to image and formula

        Args:
            sample: tuple(image path, formula id)

        Returns:
            img(ndarray): actual image data
            formula:
        """
        img_path, formula_id = sample
        inst = None

        img = imread(self._dir_images + "/" + img_path)
        img = self._img_pre(img)
        formula = self._formula_pre(self._get_raw_formula(formula_id))

        if self._iter_mode == 'data':
            inst = (img, formula)
        elif self._iter_mode == 'full':
            inst = (img, formula, img_path, formula_id)

        # filter on the formula length
        if self._max_len is not None and len(formula) > self._max_len:
            skip = True
        else:
            skip = False

        return inst, skip

    def __iter__(self):
        """Iterator over Dataset

        Yields:
            tuple (img, formula)
        """
        n_iter = 0
        for sample in self._data_generator:
            if self._max_iter is not None and n_iter >= self._max_iter:
                break
            inst, skip = self._process_instance(sample)
            if skip:
                continue
            n_iter += 1
            yield inst

    def bucket(self, bucket_size):
        """Iterates over the listing and creates buckets of same shape images

        Args:
            bucket_size(int): size of the bucket

        Returns:
            bucketed_dataset(list[tuple]): [(img_path1, id1), ...]
        """
        print("Bucketing the dataset ...")
        bucketed_dataset = []

        old_mode = self._iter_mode # store the old iteration mode
        self._iter_mode = "full"

        # iterate over the dataset in "full" mode and create buckets
        data_buckets = dict() # buffer for buckets
        for idx, (img, formula, img_path, formula_id) in enumerate(self):
            s = img.shape
            if s not in data_buckets:
                data_buckets[s] = []
            # if bucket is full, write it and empty it
            if len(data_buckets[s]) == bucket_size:
                for (img_path, formula_id) in data_buckets[s]:
                    bucketed_dataset += [(img_path, formula_id)]
                data_buckets[s] = []

            data_buckets[s] += [(img_path, formula_id)]

        # write the rest of the buffer
        for k, v in data_buckets.items():
            for (img_path, formula_id) in v:
                bucketed_dataset += [(img_path, formula_id)]

        self._iter_mode = old_mode
        self._length = idx + 1

        print("bucket done.")
        return bucketed_dataset

    def build(self, quality=100, density=200, down_ratio=2, buckets=None,
              n_threads=4):
        """Generates images from the formulas and writes the correspondance
        in the matching file.

        Args:
            quality: parameter for magick
            density: parameter for magick
            down_ratio: (int) downsampling ratio
            buckets: list of tuples (list of sizes) to produce similar
                shape images

        """
        # 1. produce images
        create_dir(self._dir_images)
        result = build_images(self._formulas, self._dir_images, quality,
                              density, down_ratio, buckets, n_threads)

        # 2. write matching with same convention of naming
        with open(self._path_matching, "w") as f:
            for (path_img, idx) in result:
                if path_img is not False:  # image was successfully produced
                    f.write("{} {}\n".format(path_img, idx))

    def minibatch(self, minibatch_size):
        """ get minibatches of dataset

        Args:
            minibatch_size(int): batch size

        Returns:
            list of tuples

        """
        x_batch, y_batch = [], []
        for (x, y) in self:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []

            x_batch += [x]
            y_batch += [y]

        if len(x_batch) != 0:
            yield x_batch, y_batch
