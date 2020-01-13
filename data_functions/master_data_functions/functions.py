import os
import numpy as np


def import_data(path=None, num_samples=None, scaling=False):
    """ Imports scintillator data as numpy arrays.
    Used together with analysis repository which has a strict folder
    structure.

    param path: Path to datafile

    param num_samples:  How many samples to include. With large files,  
                        memory might become an issue when loading full file.
                        If specified, the returned data will be a random,
                        balanced selection of data from the full dataset.

    param scaling:  Whether or not to scale the image data to 0-1 interval.
                    Defaults to True.
    

    returns:    dictionary of data where each filenames are keys and each
                key,value pair contains dictionary of the data in the file,
                separated into 'energies', 'positions', 'images', 'labels'.
    """

    # Get individual file paths and filenames from data folder
    # Load each datafile into a dictionary, using the filenames as keys
    # Each key then contains a dict with keys 'energies', 'positions'
    # 'images', and 'labels'

    separated_data = {}

    # Temporary initialization of arrays-to-be
    images = []
    energies = []
    positions = []
    labels = []

    # read line by line to alleviate memory strain when files are large
    with open(path, "r") as infile:
        for line in infile:
            line = np.fromstring(line, sep=' ')
            image, energy, position = separate_simulated_data(line)
            label = label_simulated_data(line)

            images.append(image)
            energies.append(energy)
            positions.append(position)
            labels.append(label)

    # Convert lists to numpy arrays and reshape them to remove the added axis from
    # conversion. TODO: Find a better way to do this?
    images = np.array(images)
    energies = np.array(energies)
    positions = np.array(positions)
    labels = np.array(labels)

    images = images.reshape(images.shape[0], images.shape[2], images.shape[3], images.shape[4])
    energies = energies.reshape(energies.shape[0], energies.shape[2])
    positions = positions.reshape(positions.shape[0], positions.shape[2])


    # Pick a num_samples randomly selected samples such that the returned
    # dataset contains a balanced number of single and double events.
    if num_samples is not None and int(num_samples) < len(images):

        # Convert to int so number can be provided on scientific form ex. 1e5
        num_samples = int(num_samples)

        # Get separate indices for single and double events based on labels
        single_indices = np.array(np.where(labels == 0)[0])
        double_indices = np.array(np.where(labels == 1)[0])

        # Handle cases where number of samples is not an even number
        if num_samples % 2 != 0:
            num_double = num_samples//2
            num_single = num_double + 1
        else:
            num_single = num_double = num_samples//2

        # Handle cases where dataset contains fewer than num_samples/2 of
        # an event type
        if len(single_indices) < num_single:
            num_single = len(single_indices)
        if len(double_indices) < num_double:
            num_double = len(double_indices)

        # Draw random indices single and double indices
        single_out = np.random.choice(single_indices, size=num_single, replace=False)
        double_out = np.random.choice(double_indices, size=num_double, replace=False)

        # Concatenate the selected images, energies, positions and labels.
        images = np.concatenate((images[single_out], images[double_out]), axis=0)
        energies = np.concatenate((energies[single_out], energies[double_out]), axis=0)
        positions = np.concatenate((positions[single_out], positions[double_out]), axis=0)
        labels = np.concatenate((labels[single_out], labels[double_out]), axis=0)

    # Store the data for return
    separated_data["images"] = images
    separated_data["energies"] = energies
    separated_data["positions"] = positions
    separated_data["labels"] = labels
    
    return separated_data

def separate_simulated_data(data):
    """Takes an imported dataset and separates it into images, energies 
    and positions.

    Data info:
    The first 256 values in each row correspond to the 16x16 detector image and
    the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2, Ypos2.

    param data: array of one or more lines from datafile
    
        
    returns: list, [images, energies, positions]
    """

    # If data is just one line, reshape to (1, data.shape)
    if len(data.shape) < 2:
        data = data.reshape((1, len(data)))
    
    n_pixels = data.shape[1] - 6 #account for 6 non-pixel values
    n_img = data.shape[0] # Number of sample images

    # reshape to image dims (batch, rows, cols, channels)
    images = data[:, :n_pixels].reshape(n_img, 16, 16, 1)
    # transpose to correct spatial orientation
    images = np.transpose(images, axes=[0, 2, 1, 3])
    
    # Extract energies and positions as array with columns [Energy1, Energy2]
    # and positions array with columns [Xpos1, Ypos1, Xpos2, Ypos2]
    energy1 = data[:, n_pixels].reshape(n_img, 1) # reshape for stacking
    energy2 = data[:, n_pixels+3].reshape(n_img, 1) # reshape for stacking
    energies = np.hstack((energy1, energy2))

    position1 = data[:, n_pixels+1:n_pixels+3]
    position2 = data[:, n_pixels+4:]
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
        data = data.reshape((1,)+data.shape)
    n_samples = data.shape[0]
    n_pixels = 256 # The images are a 16x16 grid, flattened
    energy1 = data[:, n_pixels].reshape(n_samples, 1) # reshape for stacking
    energy2 = data[:, n_pixels+3].reshape(n_samples, 1) # reshape for stacking
    energies = np.hstack((energy1, energy2))

    labels = np.where(energies[:,1] != 0, 1, 0)
            
    return labels

def normalize_image_data(images):
    """ Takes an imported set of images and normalizes values to between
    0 and 1 using min-max scaling across the whole image set.
    """
    if len(image.shape) == 4:
        img_term = np.amax(images[:,1:3]) - np.amin(images[:,1:3])
        img_mean = np.mean(images[:,1:3])
        images[:,1:3] = (images[:,1:3] - img_mean) / img_term
    else:
        img_term = np.amax(images) - np.amin(images)
        img_mean = np.mean(images)
        images = (images - img_mean) / img_term
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
    np.save(OUTPUT_PATH+filename, features)

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
    return np.load(OUTPUT_PATH+filename)

def save_model(filename):
    raise NotImplemented

def load_model(filename):
    raise NotImplemented

def event_indices(positions, threshold=3.0):
    """ Returns indices of events with a distance lower than a certain 
    threshold to do further training on.

    param positions:    array of positions (x0, y0, x1, y1)
    param threshold:    float, the threshold which determines what is a
                        'close' event.

    returns:    Indices for single events, double events, and for the subset 
                of double events which are 'close' events.
    """
    indices_single = np.where(positions[:, 2] < 0)[0]
    indices_double = np.where(positions[:, 2] >= 0)[0]
    dist = relative_distance(positions)
    indices_close = np.nonzero((dist >= 0) == (dist < threshold))[0]
    
    return indices_single, indices_double, indices_close

def relative_distance(positions):
    """ Calculates the relative distance between events in a set of events.
    
    param positions: Array of positions for the dataset (x0, y0, x1, y1)

    return: 1D array of relative distances between events.
            single events have relative distance set to -100
    """
   
    # Single events have the x1, y1 positions set to -100. We don't want to
    # do anything about those and simply set the relative distance to -100.
    single_indices = np.where(positions[:, 2] < 0)[0]
    double_indices = np.where(positions[:, 2] >= 0)[0]
   
    relative_dist = np.zeros((positions.shape[0], 1))
    relative_dist[single_indices] = -100

    # Standard euclidian distance between points 
    # np.sqrt((x0-x1)**2 + (y0-y1)**2)
    # We also multiply by 3 to scale the distance from pixels to mm
    relative_dist[double_indices] = np.sqrt(np.sum(
            (positions[double_indices, 0:2] - positions[double_indices, 2:])**2 * 3,
            axis=1)).reshape(len(double_indices), 1)

    return relative_dist


def relative_energy(energies):
    """ Calculates the relative energy E1/E2 between event 1 and event 2 for all
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
   
    relative_energies = np.zeros((energies.shape[0], 1))
    relative_energies[single_indices] = -100

    relative_energies[double_indices] = np.reshape(
            np.amin(energies[double_indices], axis=1) / np.amax(energies[double_indices], axis=1), 
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
            energies[double_indices, 0] - energies[double_indices, 1]).reshape(len(double_indices), 1)
    
    return energy_differences

