import numpy as np


def separate_simulated_data(data):    
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
    
    # Extract energies and positions as array with columns [Energy1, Energy2]
    # and positions array with columns [Xpos1, Ypos1, Xpos2, Ypos2]
    energy1 = data[:, n_pixels].reshape(n_img, 1) # reshape for stacking
    energy2 = data[:, n_pixels+3].reshape(n_img, 1) # reshape for stacking
    energies = np.hstack((energy1, energy2))

    position1 = data[:, n_pixels+1:n_pixels+3]
    position2 = data[:, n_pixels+4:]
    positions = np.hstack((position1, position2))
    
    return [images, energies, positions]

def label_simulated_data(energies):
    """Given arrays of energies, produces a set of labels for the dataset 
    for use in classification with
    0 -> single event
    1 -> double event 
    The labels are determined from the energy values, as we know that if 
    there is no second particle then Energy2 = 0."""
    
    n_samples = energies.shape[0]
    #n_classes = energies.shape[1]

    #labels = np.zeros((n_samples))
    # returns an n_samples x n_classes array where if the condition is true the value is 1
    # and if the condition is false the value is 0, corresponding to
    # double and single events, respectively.
    # np.where(condition, return if true, return if false)
    labels = np.where(energies[:,1] != 0, 1, 0)
    #labels[:,1] = np.where(energies[:,1] != 0, 1, 0)
            
    return labels
