from utils.config import Config


def test_config(config_dir_name):
    config_fromfile = Config(config_dir_name)
    for k, v in config_fromfile.__dict__.items():
        print("{}: {}".format(k, v))

    config_fromdict = Config({"learning_rate": 0.001})
    for k, v in config_fromdict.__dict__.items():
        print("{}: {}".format(k, v))

    config_fromlist = Config([config_dir_name])
    for k, v in config_fromlist.__dict__.items():
        print("{}: {}".format(k, v))

    config_fromfile.save("./")

    return 0


if __name__ == '__main__':
    test_config("../config/config_test.json")
