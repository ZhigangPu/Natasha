from inputer.data_generator import DataGenerator
from utils.utils import get_logger


def main():
    path_formulas = "../data/train.formulas.norm.txt"
    dir_images = "../data/images_train/"
    path_matching = "../data/train.matching.txt"

    train_set = DataGenerator(
        path_formulas=path_formulas,
        dir_images=dir_images,
        path_matching=path_matching,
        max_iter=10
    )

    for sample in train_set.minibatch(9):
        image, formula = sample


if __name__ == '__main__':
    main()


