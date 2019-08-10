# Scripts
Scripts used in relation to my master's thesis.

## data_functions/
Contains functions used to import data from scintillator experiments,\
either real or simulated. Structured as a python module to make importing\
easier across notebooks etc.

### Information about file format
For all data files each image and label is on one row.\
The first 256 values in each row correspond to the 16x16 detector image and \
the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2, Ypos2.\  
If there is no second particle then Energy2 = 0 and Xpos2 and Ypos2 are both -100.\  
(When I run my model, I have to reset the -100 to 0).\

