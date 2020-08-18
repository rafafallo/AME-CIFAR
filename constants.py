# Directory where all results are stored.
run_path = './runs'
idx_digits = 3

def filename(s, idx = None, e = ''):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    if idx is None:
        return run_path + '/' + s + e
    else:
        return run_path + '/' + s + '-' + str(idx).zfill(3) + e


def csv_filename(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, i, '.csv')


def data_filename(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, i, '.npy')


def picture_filename(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, i, '.png')


# Models (neural networks).
full_model_filename = filename('full-model')
features_model_filename = filename('features-model')

# Features and labels.
train_features_filename = filename('train_features.npy')
test_features_filename = filename('test_features.npy')
train_labels_filename = filename('train_labels.npy')
test_labels_filename = filename('test_labels.npy')

n_memory_tests = 1
n_objects = 10
objects_per_memory = [0, 1, 2]

all_labels = list(range(n_objects))
  
precision_idx = 0
recall_idx = 1
n_measures = 2

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
domain = 640