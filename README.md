## Coding Project for the course Advanced Scientific Programming with Python that was offered in in March 2019.
### This project is based on the code I developed for extracting the signal from biosensors as part of my doctoral research. The code developed for this code does the following tasks:
#### 1) Import the input data file and do some basic noise reduction.

#### 2) Segregate the data into two groups: high and low arrays to separate these two values of current.

#### 3) Replace each individual pulse by an average over the the values of that particular pulse, see 3_average_over_each_pulse.png.

#### 4) Extract the signal by subtracting the low current from the high current after using piecewise interpolation to fill up the alternating gaps in the data.

#### 5) Finally subtract the low from the high current to obtain the signal.


