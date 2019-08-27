import os
import numpy as np


def import_data(folder='sample', filename=None, num_samples=None, scaling=True):
    """ Imports scintillator data as numpy arrays.
    Used together with analysis repository which has a strict folder
    structure.

    param folder:   Which data to load. Defaults to 'sample'.
        'real'      ->  real data from scintillator experiment
        'sample'    ->  sample data (simulated)
        'simulated' ->  simulated data from scintillator

    param filename: Which file to load. If not provided, attempts to load
                    all files in folder

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

    # Verify input and set DATA_PATH
    if folder == 'real':
        DATA_PATH = '../../data/real/'
    elif folder == 'sample':
        DATA_PATH = '../../data/sample/'
    elif folder == 'simulated':
        DATA_PATH = '../../data/simulated/'
    else:
        print("Invalid data folder specified.") 
        print("Must be 'real', 'sample', or 'simulated'")
        exit(1)

    # Get individual file paths and filenames from data folder
    # Load each datafile into a dictionary, using the filenames as keys
    # Each key then contains a dict with keys 'energies', 'positions'
    # 'images', and 'labels'
    data = {}

    if filename is None:
        #import the whole dir
        data_dir = os.scandir(DATA_PATH)
        for FILE_PATH in data_dir:
            tmp_filename = FILE_PATH.name

            # Skip this file if it's the README file
            if "README" in tmp_filename:
                continue

            # Load data
            full_data = np.loadtxt(FILE_PATH.path)
            separated_data = {}
            
            # Currently uncertain if real data has same form as simulated.
            # If so, the if-test here can be discarded, and functions for
            # simulated data can be renamed to something more generic.

            if folder in ['simulated', 'sample']:
                images, energies, positions = separate_simulated_data(full_data, scaling)
                labels = label_simulated_data(full_data)

                separated_data["images"] = images
                separated_data["energies"] = energies
                separated_data["positions"] = positions
                separated_data["labels"] = labels

            data[tmp_filename] = separated_data
    else:
        separated_data = {}

        # Temporary initialization of arrays-to-be
        images = []
        energies = []
        positions = []
        labels = []

        # read line by line to alleviate memory strain when files are large
        with open(DATA_PATH+filename, "r") as infile:
            for line in infile:
                line = np.fromstring(line, sep=' ')
                image, energy, position = separate_simulated_data(line, scaling)
                label = label_simulated_data(line)

                images.append(image)
                energies.append(energy)
                positions.append(position)
                labels.append(label)

        # Convert lists to numpy arrays and reshape them to remove the added axis from
        # conversion. TODO: Find a better way to do this.
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
        
        data[filename] = separated_data

    return data

def separate_simulated_data(data, scaling):    
    """Takes an imported dataset and separates it into images, energies 
    and positions.

    Data info:
    The first 256 values in each row correspond to the 16x16 detector image and
    the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2, Ypos2.

    param data: array of one or more lines from datafile
    param scaling: Boolean, scale images to 0-1 or not.
    
        
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
    if scaling:
        images = normalize_image_data(images)
    
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
    x_max = np.amax(images)
    x_min = np.amin(images)
    images = (images - np.mean(images)) / (x_max - x_min)
    return images

def save_feature_representation(features, filename):
    """ Takes a set of data represented as features (after being) fed
    though a trained network, and saves it as a numpy object.

    param feature: The features to be saved.

    """

    OUTPUT_PATH = '../../data/output/'
    np.save(OUTPUT_PATH+filename, features)

def load_feature_representation(filename):
    """ Given a filename, load a numpy file object from the output folder
    matching the filename.
    """
    OUTPUT_PATH = '../../data/output/'
    return np.load(OUTPUT_PATH+filename)
