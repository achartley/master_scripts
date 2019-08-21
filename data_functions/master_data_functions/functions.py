import os
import numpy as np


def import_data(folder='sample', scaling=True):
    """ Imports scintillator data as numpy arrays.
    Used together with analysis repository which has a strict folder
    structure.

    param folder:   Which data to load. Defaults to 'sample'.
        'real'      ->  real data from scintillator experiment
        'sample'    ->  sample data (simulated)
        'simulated' ->  simulated data from scintillator
                    

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

    data_dir = os.scandir(DATA_PATH)
    for FILE_PATH in data_dir:
        filename = os.path.splitext(FILE_PATH.name)[0] # Don't need file extension

        # Skip this file if it's the README file
        if "README" in filename:
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

        data[filename] = separated_data

    return data
    
def separate_simulated_data(data, scaling):    
    """Takes an imported dataset and separates it into images, energies 
    and positions. Could potientially be expanded to return e.g a Pandas
    datafram if useful.
    
    Datafile info:
    The first 256 values in each row correspond to the 16x16 detector image and
    the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2, Ypos2.
        
    returns: list, [images, energies, positions]
    """
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

