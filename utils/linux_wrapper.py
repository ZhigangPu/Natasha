import os
import subprocess


def run(cmd, timeout_sec):
    """Run cmd in the shell with timeout"""
    proc = subprocess.Popen(cmd, shell=True)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def create_dir(dir_name):
    """Creates directory if it does not exsit"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def init_file(path_file, mode="a"):
    """Makes sure that a given file exists"""
    with open(path_file, mode) as f:
        pass


def get_files(dir_name):
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    return files


def delete_file(path_file):
    try:
        os.remove(path_file)
    except Exception:
        pass
