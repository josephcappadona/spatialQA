import sys, os

def configure_path(cur_file):
    sys.path.append(
        os.path.join(os.path.realpath(os.path.dirname(cur_file)),
                     os.pardir,
                     'src'))