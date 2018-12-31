import os
import h5py
import data
from data_util import config


def bin_files_to_h5py(data_path, fileType):
    numExamples = 0
    examples = data.text_generator(data.example_generator(data_path, True, False))

    for _ in examples:
        numExamples += 1

    print(f"Counted number of examples: {numExamples}")
    examples = data.text_generator(data.example_generator(data_path, True, False))

    # Next write to file
    db_path = h5_db_filepath(fileType)
    db = h5py.File(db_path,
                   mode="w")
    str_dtype = h5py.special_dtype(vlen=str)
    articles = db.create_dataset("articles", (numExamples,), dtype=str_dtype)
    abstracts = db.create_dataset("abstracts", (numExamples,), dtype=str_dtype)

    for i, (article, abstract) in enumerate(examples):
        articles[i] = article
        abstracts[i] = abstract

    db.close()


def h5_db_filepath(file_type):
    return os.path.join(config.root_dir, f"cnn-dailymail-master/finished_files/{file_type}.hdf5")


def check_if_h5_db_exists(file_type):
    path = h5_db_filepath(file_type)
    return os.path.isfile(path)


def convert_bin_files_to_h5():
    bin_paths = {"train": config.train_data_path, "val": config.eval_data_path, "test": config.decode_data_path}

    for file_type, data_path in bin_paths.items():
        if not check_if_h5_db_exists(file_type):
            print(f"Creating {file_type} file")
            bin_files_to_h5py(data_path, file_type)
        else:
            print(f"{file_type} H5 DB already exists")


if __name__ == "__main__":
    convert_bin_files_to_h5()
    # print("Creating train file")
    # bin_files_to_h5py(config.train_data_path, "train")
    # print("Creating validation file")
    # bin_files_to_h5py(config.eval_data_path, "val")
    # print("Creating test file")
    # bin_files_to_h5py(config.decode_data_path, "test")

