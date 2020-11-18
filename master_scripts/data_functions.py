import json
import re
import git
import numpy as np
import tensorflow as tf
import pandas as pd
import subprocess
from datetime import datetime
from sklearn.model_selection import train_test_split


def get_git_root():
    """ Get the git repo root path to use relative file paths
    in the rest of the code.
    """

    rpath = git.Repo('.', search_parent_directories=True).working_tree_dir
    rpath = rpath + '/'
    return rpath


def get_tf_device(MAX_LOAD):
    """Determine tensorflow device.
    Checks for GPU availability given MAX_LOAD and sets DEVICE
    to use in training or prediction
    """
    DEVICE = None

    # Set memory growth and list logical devices
    if int(tf.__version__[0]) < 2:
        physical_gpu = tf.config.experimental.list_physical_devices('GPU')
        for gpu in physical_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpu_devices = tf.config.experimental.list_logical_devices('GPU')
        cpu_devices = tf.config.experimental.list_logical_devices('CPU')
    else:
        physical_gpu = tf.config.list_physical_devices('GPU')
        for gpu in physical_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpu_devices = tf.config.list_logical_devices('GPU')
        cpu_devices = tf.config.list_logical_devices('CPU')

    # Select suitable GPU or default to CPU
    if gpu_devices:
        nvidia_command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv"]
        nvidia_output = subprocess.run(
            nvidia_command, text=True, capture_output=True).stdout
        gpu_loads = np.array(re.findall(r"(\d+), (\d+) %", nvidia_output),
                             dtype=np.int)  # tuple (id, load%)
        eligible_gpu = np.where(gpu_loads[:, 1] < MAX_LOAD)
        if len(eligible_gpu[0]) == 0:
            print("No GPUs with less than 20% load. Check nvidia-smi.")
            exit(0)
        else:
            # Choose the highest id eligible GPU
            # Assuming a lot of people use default allocation which is
            # lowest id.
            gpu_id = np.amax(gpu_loads[eligible_gpu, 0])
            DEVICE = gpu_devices[gpu_id].name
            print("CHOSEN GPU IS:", DEVICE)
    else:
        # Default to CPU
        DEVICE = cpu_devices[0].name
        print("NO GPU FOUND, DEFAULTING TO CPU.")

    return DEVICE


def generate_dataset_simulated(path, num_samples=None, random_state=None,
                               test_size=None):
    """Generate numpy-format dataset from a scintillator datafile.

    Keyword arguments:
    path -- full path to scintillator datafile
    num_samples -- number of samples. If None, use the whole file.
    random_state -- set random seed for reproducibility
    test_size -- proportion of the dataset to include in the test data.
                 E.g 0.1 splits off 10% of the data as test data.
    """

    repo_root = get_git_root()
    print("Importing data from", path)
    print("This may take some time.")
    images, energies, positions, labels = import_data(path)

    # Set indices for train and validation
    x_idx = np.arange(images.shape[0])
    train_idx, test_idx = train_test_split(
        x_idx,
        test_size=test_size,
        random_state=random_state,
    )

    # Set filenames
    if test_size is not None:
        training_len = int(images.shape[0] * (1.0 - test_size))
    else:
        training_len = int(images.shape[0] * 0.75)

    test_len = int(images.shape[0] - training_len)

    data_path = repo_root + "data/simulated/"
    dt_string = datetime.today().strftime('%Y%m%d%H%M')
    training_set_name = "training_" + str(training_len) + "_" + dt_string
    test_set_name = "test_" + str(test_len) + "_" + dt_string
    print("Writing to numpy format...")
    print("{images, energies, positions, labels}_" + training_set_name)
    print("{images, energies, positions, labels}_" + test_set_name)

    # Save training files:
    np.save(data_path + "images_" + training_set_name, images[train_idx])
    np.save(data_path + "energies_" + training_set_name, energies[train_idx])
    np.save(data_path + "positions_" + training_set_name, positions[train_idx])
    np.save(data_path + "labels_" + training_set_name, labels[train_idx])

    # Save test files:
    np.save(data_path + "images_" + test_set_name, images[test_idx])
    np.save(data_path + "energies_" + test_set_name, energies[test_idx])
    np.save(data_path + "positions_" + test_set_name, positions[test_idx])
    np.save(data_path + "labels_" + test_set_name, labels[test_idx])


def import_data(path):
    """ Imports scintillator data as numpy arrays.
    Used together with analysis repository which has a strict folder
    structure.

    param path: Path to datafile

    returns:    dictionary of data where each filenames are keys and each
                key,value pair contains dictionary of the data in the file,
                separated into 'energies', 'positions', 'images', 'labels'.
    """

    # Temporary initialization of arrays-to-be
    images = []
    energies = []
    positions = []
    labels = []

    # Read line by line to alleviate memory strain when files are large
    # The first 256 values in each row correspond to the 16x16 detector image,
    # the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2,
    # Ypos2.

    with open(path, "r") as infile:
        for line in infile:
            line = np.fromstring(line, sep=' ')
            image = line[:256]
            energy = np.array((line[256], line[259]))
            pos = np.array((line[257], line[258], line[260], line[261]))

            # Set label for the events. If Pos[3] is -100 it is a single
            # event. Any other values corresponds to a double event.
            # We label single events as type 0, and doubles as type 1
            if pos[3] == -100:
                label = 0
            else:
                label = 1

            images.append(image)
            energies.append(energy)
            positions.append(pos)
            labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    energies = np.array(energies)
    positions = np.array(positions)
    labels = np.array(labels)

    # Reshape and transpose images to align positions with spatial
    # orientation of images
    images = images.reshape((images.shape[0], 16, 16))
    images = np.transpose(images, (0, 2, 1))
    return images, energies, positions, labels


def import_real_data(path, num_samples=None, return_events=True):
    """ Imports experimental data as numpy arrays.
    Used together with analysis repository which has a strict folder
    structure.

    param path: config containing paths, modelnames etc.

    param num_samples:  How many samples to include. With large files,
                        memory might become an issue when loading full file.
                        If specified, the returned data will be the first
                        n samples


    returns: dictionary of: { event_id: {event_descriptor: int,
                                         image: array,
                                        }
    """

    # Dictionary for storing events
    events = {}
    images = []
    # read line by line to alleviate memory strain when files are large
    with open(path, "r") as infile:
        image_idx = 0
        for line in infile:
            # If we have the desired amount of samples, return events.
            if num_samples and len(events.keys()) == num_samples:
                return events

            line = np.fromstring(line, sep=' ')
            event_id = int(line[0])
            event_descriptor = int(line[1])
            image = np.array(line[2:], dtype=np.float32).reshape((16, 16, 1))
            images.append(image)
            events[event_id] = {
                "event_descriptor": event_descriptor,
                "image_idx": image_idx
            }
            image_idx += 1

    images = np.array(images)
    # images = np.transpose(images, (0, 2, 1, 3))
    if return_events:
        return events, images
    else:
        return images


def separate_simulated_data(data):
    """Takes an imported dataset and separates it into images, energies
    and positions.

    Data info:
    The first 256 values in each row correspond to the 16x16 detector image and
    the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2,
    Ypos2.

    param data: array of one or more lines from datafile


    returns: list, [images, energies, positions]
    """

    # If data is just one line, reshape to (1, data.shape)
    if len(data.shape) < 2:
        data = data.reshape((1, len(data)))

    n_pixels = data.shape[1] - 6  # account for 6 non-pixel values
    n_img = data.shape[0]  # Number of sample images

    # reshape to image dims (batch, rows, cols, channels)
    images = data[:, :n_pixels].reshape(n_img, 16, 16, 1)

    # Extract energies and positions as array with columns [Energy1, Energy2]
    # and positions array with columns [Xpos1, Ypos1, Xpos2, Ypos2]
    energy1 = data[:, n_pixels].reshape(n_img, 1)  # reshape for stacking
    energy2 = data[:, n_pixels + 3].reshape(n_img, 1)  # reshape for stacking
    energies = np.hstack((energy1, energy2))

    position1 = data[:, n_pixels + 1:n_pixels + 3]
    position2 = data[:, n_pixels + 4:]
    positions = np.hstack((position1, position2))

    return [images, energies, positions]


def label_simulated_data(data):
    """Given arrays of energies, produces a set of labels for the dataset
    for use in classification with
    0 -> single event
    1 -> double event
    The labels are determined from the energy values, as we know that if
    there is no second particle then Energy2 = 0.
    """

    # Extract energies as array with columns [Energy1, Energy2]
    if len(data.shape) < 2:
        data = data.reshape((1,) + data.shape)
    n_samples = data.shape[0]
    n_pixels = 256  # The images are a 16x16 grid, flattened
    energy1 = data[:, n_pixels].reshape(n_samples, 1)  # reshape for stacking
    # reshape for stacking
    energy2 = data[:, n_pixels + 3].reshape(n_samples, 1)
    energies = np.hstack((energy1, energy2))

    labels = np.where(energies[:, 1] != 0, 1, 0)

    return labels


def normalize_image_data(images):
    """ Takes an imported set of images and normalizes values to between
    0 and 1 using min-max scaling across the whole image set.
    """
    img_max = np.amax(images)
    img_min = np.amin(images)
    images = (images - img_min) / (img_max - img_min)
    return images


def normalize_image_data_elementwise(images):
    """ Takes an imported set of images and normalizes values to between
    0 and 1 using min-max scaling on each image
    """
    img_max = np.amax(images.reshape(images.shape[0], 256), axis=1)
    img_min = np.amin(images.reshape(images.shape[0], 256), axis=1)
    term_top = images - img_min.reshape(images.shape[0], 1, 1, 1)
    term_bottom = (img_max - img_min).reshape(images.shape[0], 1, 1, 1)
    images = term_top / term_bottom
    return images


def normalize_position_data(positions):
    """ Takes an imported set of positions and normalizes values to between
    0 and 1 simply dividing by 16.
    """
    single_indices, double_indices, close_indices = event_indices(positions)
    positions[single_indices, 2:] = -1.0
    positions[single_indices, :2] /= 16
    positions[double_indices] /= 16
    return positions


def save_feature_representation(filename, features, path=None):
    """ Takes a set of data represented as features (after being) fed
    though a trained network, and saves it as a numpy object.

    param path: directory to save features in.
    param filename: filename to save the features to.
    param feature: The features to be saved.
    """
    if path:
        OUTPUT_PATH = path
    else:
        OUTPUT_PATH = '../../data/simulated/'
    np.save(OUTPUT_PATH + filename, features)


def load_feature_representation(filename, path=None):
    """ Given a filename, load a numpy file object from the output folder
    matching the filename.
    param path: directory to load features from.
    param filename: filename of features to be loaded.
    """
    if path:
        OUTPUT_PATH = path
    else:
        OUTPUT_PATH = '../../data/simulated/'
    return np.load(OUTPUT_PATH + filename)


def event_indices(positions, threshold=1.0):
    """ Returns indices of events with a distance lower than a certain
    threshold to do further training on.

    :param positions:    array of positions (x0, y0, x1, y1)
    :param threshold:    float, the threshold which determines what is a
                        'close' event. 1.0 corresponds to 1 pixel

    :returns:    Indices for single events, double events, and for the subset
                of double events which are 'close' events.
    """
    indices_single = np.where(positions[:, 2] < 0)[0]
    indices_double = np.where(positions[:, 2] >= 0)[0]
    dist = separation_distance(positions)
    indices_close = np.nonzero((dist >= 0) == (dist < threshold))[0]

    return indices_single, indices_double, indices_close


def separation_distance(positions):
    """ Calculates the separation distance (in pixels) between double events
    in a set of events.

    param positions: Array of positions for the dataset (x0, y0, x1, y1)

    return: 1D array of separation distances between events.
            single events have separation distance set to -100
    """

    # Single events have the x1, y1 positions set to -100. We don't want to
    # do anything about those and simply set the separation distance to -100.
    single_indices = np.where(positions[:, 2] < 0)[0]
    double_indices = np.where(positions[:, 2] >= 0)[0]

    separation_dist = np.zeros((positions.shape[0], 1))
    separation_dist[single_indices] = -100

    # Standard euclidian distance between points
    separation_dist[double_indices] = np.sqrt(np.sum(
        (positions[double_indices, 0:2]
         - positions[double_indices, 2:])**2,
        axis=1)).reshape(len(double_indices), 1)

    return separation_dist


def relative_energy(energies, scale=False):
    """ Calculates the relative energy E1/E2 between event 1 and event 2 for all
    samples in a dataset which have two events.

    param energies: Energies of events in the dataset, (E0, E1).
                    For single events the energy of 'event 2' is set to
                    0 in the dataset.

    param scale:    Boolean, defaults to False. If True, scales the relative
                    energies to [0, 1] by always dividing argmin(E1, E2) by
                    argmax(E1, E2).

    return: 1D array of relative energies. single events have relative energy
            set to -100.
    """

    # Single events have the E2 energy to 0. We don't want to
    # do anything about those and simply set the relative energy to -100.
    # (There is a chance a double event has same energy and thus gives
    # relative energy = 0, thus -100 is a safer choice of single event default)
    double_indices = np.where(energies[:, 1] != 0)[0]
    single_indices = np.where(energies[:, 1] == 0)[0]

    relative_energies = np.zeros((energies.shape[0], 1))
    relative_energies[single_indices] = -100

    if scale:
        # Divide amin(E1,E2) / amin(E1, E2)
        relative_energies[double_indices] = np.reshape(
            np.amin(energies[double_indices], axis=1)
            / np.amax(energies[double_indices], axis=1),
            (len(double_indices), 1)
        )
    else:
        # Divide E1/E2 for all events.
        relative_energies[double_indices] = np.reshape(
            energies[double_indices, 0] / energies[double_indices, 1],
            (len(double_indices), 1)
        )

    return relative_energies


def energy_difference(energies):
    """ Calculates the energy difference between event 1 and event 2 for all
    samples in a dataset which have two events.

    param energies: Energies of events in the dataset, (E0, E1). For single
                    events the energy of 'event 2' is set to 0 in the dataset.

    return: 1D array of relative energies. single events have relative energy
            set to -100.
    """

    # Single events have the E2 energy to 0. We don't want to
    # do anything about those and simply set the relative energy to -100.
    # (There is a chance a double event has same energy and thus gives
    # relative energy = 0, thus -100 is a safer choice of single event default)
    double_indices = np.where(energies[:, 1] != 0)[0]
    single_indices = np.where(energies[:, 1] == 0)[0]

    energy_differences = np.zeros((energies.shape[0], 1))
    energy_differences[single_indices] = -100

    energy_differences[double_indices] = np.abs(
        energies[double_indices, 0] - energies[double_indices, 1]
    ).reshape(len(double_indices), 1)

    return energy_differences


def load_experiment(e_id):
    repo_root = get_git_root()
    e_path = repo_root + "experiments/"
    with open(e_path + e_id + ".json", "r") as fp:
        e = json.load(fp)
    return e


def load_hparam_search(name):
    """ Reads json-formatted hparam search file spec to pandas DF,
    and loads additional metrics into the dataframe.
    """
    repo_root = get_git_root()
    hpath = repo_root + "experiments/searches/"
    df = pd.read_json(
        hpath + name, orient='index').rename_axis('id').reset_index()
    # JSON convert the tuples in hparam search to list when it's interpreted.
    # Convert the values to str to make it workable
    df['kernel_size'] = [str(x) for x in df['kernel_size'].values]
    # Add additional metrics to df
    accs = []
    f1 = []
    mcc = []
    auc = []
    for e_id in df['id']:
        e = load_experiment(e_id)
        accs.append(e['metrics']['accuracy_score'])
        f1.append(e['metrics']['f1_score'])
        mcc.append(e['metrics']['matthews_corrcoef'])
        auc.append(e['metrics']['roc_auc_score'])
    df['accuracy_score'] = accs
    df['f1_score'] = f1
    df['matthews_corrcoef'] = mcc
    df['roc_auc_score'] = auc
    return df
