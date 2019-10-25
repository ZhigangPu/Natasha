import logging


def get_logger(filename):
    """Return instance of logger"""
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def load_formulas(filename):
    """load formulas form file

    Args:
        filename(str): path to formulas file

    Return:
        formulas(dict): index -> raw_formula(str)
    """
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()

    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas
