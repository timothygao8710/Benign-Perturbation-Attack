import pickle

def dump_data(data, file_path):
    """
    Serializes `data` to a file specified by `file_path` using pickle.

    Parameters:
    data (any): The data to be serialized and written to the file.
    file_path (str): The path to the file where the data should be written.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"Error occurred while dumping data to {file_path}: {e}")

def load_data(file_path):
    """
    Deserializes data from a file specified by `file_path` using pickle.

    Parameters:
    file_path (str): The path to the file from which the data should be read.

    Returns:
    any: The data deserialized from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Error occurred while loading data from {file_path}: {e}")
        return None
