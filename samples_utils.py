from os import listdir, path
from os.path import isfile, join


def file_names(path):
    return ( f for f in listdir(path) if isfile(join(path, f)) )


def change_filename_ext(filename, new_ext):
    name, ext = path.splitext(filename)
    return name + '.' + new_ext


def gen_image_part_name(image_name, num, new_ext):
    name, ext = path.splitext(image_name)
    return name + str(num) + '.' + new_ext