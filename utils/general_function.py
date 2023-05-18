import os

def get_file_list_from_dir(datadir, filetype = '.wav'):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith(filetype), all_files))
    return data_files