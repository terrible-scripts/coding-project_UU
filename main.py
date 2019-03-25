import os
os.system("cls")
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

# # Profiler start
# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
# pr.enable()

# # Timer start
# import time
# start_time = time.time()

def dataImport(inputCol1, inputCol2, startRow, delim, fileName):
    """ Function to import the two relevant columns from the input file. Possible to specify the delimiter used, and the starting row of 
    the data (after the header). """

    timeIn=[]       # stores the time data
    currentIn = []  # stores the current data
    data = np.genfromtxt(fileName, delimiter=delim)
    data = np.array(data)
    timeIn = data[startRow::,inputCol1]/60.0        # converts input time into minutes
    currentIn = data[startRow::,inputCol2]*10**12   # converts input current into pico-amperes
    timeIn = np.array(timeIn, dtype=np.float64)
    currentIn = np.array(currentIn, dtype=np.float64)
    return (timeIn, currentIn)

def dataSelector(selectRange1, selectRange2, selectRange3):
    """ The input data has 3 parts - initial and final baselines and the biomolecule binding portion. 
    Some portions of the data need to be excluded from the analysis, such as the initial drift in the baseline,
    and the disturbances introduced during the manual buffer change, or any adjustment to the system (the measurement is typically not
    stopped during this as the pause in the measurement may introduce additional drift after it is resumed). This data selection is 
    done manually in this function by using the array indices from the input data """

    timeSelect = np.concatenate([timeIn[selectRange1[0]:selectRange1[1]:], timeIn[selectRange2[0]:selectRange2[1]:], timeIn[selectRange3[0]:selectRange3[1]:]])
    currentSelect = np.concatenate([currentIn[selectRange1[0]:selectRange1[1]:], currentIn[selectRange2[0]:selectRange2[1]:], currentIn[selectRange3[0]:selectRange3[1]:]])
    return timeSelect, currentSelect

def noiseRemover(timeSelect, currentSelect, intraPulseNoise, interPulseNoise):
    """ Our experiments involve a current pulse alternating between two values of current. There are 2 major types of noise that we encounter:
    one that occurs due to the background, intraPulseNoise; and one that occurs during the switch between currents, interPulseNoise. Both these 
    known types of noise need to be minimised before signal extraction. The variables corresponding to these two types of noise represent a 
    number that must be tweaked to remove an optimal amount of noise without affecting data"""

    timeRefine = []
    currentRefine = []
    for i in range(1,len(timeSelect)-1):
        relativeForwardChange = (currentSelect[i]-currentSelect[i-1])/currentSelect[i]
        relativeBackwardChange = (currentSelect[i]-currentSelect[i+1])/currentSelect[i]
        # a data point is considered as interPulseNnoise by comparing it to the two adjacent data points
        if relativeForwardChange*relativeBackwardChange < intraPulseNoise**2 and relativeForwardChange*relativeBackwardChange > -1*interPulseNoise**2:
            timeRefine.append(timeSelect[i])
            currentRefine.append(currentSelect[i])
    timeRefine = np.array(timeRefine)
    currentRefine = np.array(currentRefine)
    return timeRefine, currentRefine

def dataSegregator(timeRefine, currentRefine, switch_s, thresh):
    """ The input data is made up of a current pulse that alternates between a lower and a higher value, and the gap between 
    these two values changes as the biomolecule binds. It is necessary to segregate the input data into a higher and a 
    lower set of values for tracing this change. This function segregates the data into 'low' and 'high' categories. This is done 
    by specifying a 'threshold' to discriminate between these types of data. This threshold is set manually after inspecting the raw data"""

    timeLow = []
    timeHigh = []
    currentLow = []
    currentHigh = []

    for i in range(0, len(timeRefine)-1):
        if(switch_s == 0):
            timeLow.append(timeRefine[i])
            currentLow.append(currentRefine[i])
        else:
            timeHigh.append(timeRefine[i])
            currentHigh.append(currentRefine[i])

        # flip the switch when the difference exceeds the threshold
        if (currentRefine[i+1]-currentRefine[i]) > thresh:
            switch_s=1
        elif (currentRefine[i+1]-currentRefine[i]) < (-1*thresh):
            switch_s=0
    return timeLow, currentLow, timeHigh, currentHigh

def pulseAveraging(time, current, pulseWidth):
    """ Onc the data has been segregated, we are left with groups of data in each set of higher and lower current values. It is necessary to 
    replace each group with an average value over that group for fitting in the following step. This produces two sets of such averaged data corresponding to the lower and 
    higher current. """

    counter=0.0
    timePulseSum = 0.0          # stores the sum of data for each pulse
    currentPulseSum = 0.0
    timeAvg = []                # stores the average of each pulse
    currentAvg = []

    for i in range(0, len(time)-1):
        if (time[i+1]-time[i]) < (pulseWidth/3.0) :
            timePulseSum += time[i]
            currentPulseSum += current[i]
            counter+=1.0
        elif counter==0.0:
            timeAvg.append(time[i])
            currentAvg.append(current[i])
        else:
            timeAvg.append(timePulseSum/counter)
            currentAvg.append(currentPulseSum/counter)
            counter=0.0
            timePulseSum = 0.0
            currentPulseSum = 0.0
    return timeAvg, currentAvg

def piecewiseNonLinearInterp(fitOrder, fitWindow, fitRange, timeLowAvg, currentLowAvg, timeHighAvg, currentHighAvg, startCheck):
    """ After obtaining the set of pulse-averaged higher and lower currents, it is necessary to obtain their difference for calculating the signal. 
    However, the two cannot be simply subtracted from each other directly. Due to a time gap between the adjacent points corresponding to the 
    lower and higher values of current, interpolation needs to be used to fill up the alternating gap in both the higher and lower currents. 
    Only then does each data point in the lower current have a corresponding point in the higher current, and vice versa, between which 
    the difference cn be taken. """
    
    timeFit = []
    currentLowFit = []                  # stores the fitted data
    currentHighFit = []
    xRange = []                         # contains the window of data to be fitted in each cycle
    yRange = []
    P = []                              # stores the fitting parameters
    init_pt = int(np.ceil(fitWindow/2)) # initial point for each piece-wise fit

    if startCheck == 0:
        for i in range(0, fitRange-fitWindow):
            # set the fitting center in the currentLow array
            timeFit.append(timeLowAvg[i+init_pt])
            currentLowFit.append(currentLowAvg[i+init_pt]) 

            # initialise/reset the fitting variables
            xRange[:] = []
            yRange[:] = []
            del P

            # store the range of currentHigh data to be fitted
            for j in range(0, fitWindow):
                xRange.append(timeHighAvg[i+j])
                yRange.append(currentHighAvg[i+j])

            # fit the currentHigh data to the specified order
            P = np.polyfit(xRange, yRange, fitOrder)
            currentHighFit.append(np.polyval(P,timeFit[len(timeFit)-1]))

            # move the fitting center to the adjacent point in the currentHigh
            currentHighFit.append(currentHighAvg[i+init_pt])
            timeFit.append(timeHighAvg[i+init_pt])

            # reset the fitting variables
            xRange[:] = []
            yRange[:] = []
            del P

            # store the range of currentLow data to be fitted
            for j in range(0, fitWindow):
                xRange.append(timeLowAvg[i+j])
                yRange.append(currentLowAvg[i+j])

            # fit the currentLow data to the specified order
            P = np.polyfit(xRange, yRange, fitOrder)
            currentLowFit.append(np.polyval(P, timeFit[len(timeFit)-1]))
    
    else:
        for i in range(0, fitRange-fitWindow):
            # set the fitting center in the currentHigh array
            timeFit.append(timeHighAvg[i+init_pt])
            currentHighFit.append(currentHighAvg[i+init_pt])

            # initialise/reset the fitting variables
            xRange[:] = []
            yRange[:] = []
            del P

            # store the range of currentLow data to be fitted
            for j in range(0, fitWindow):
                xRange.append(timeLowAvg[i+j])
                yRange.append(currentLowAvg[i+j])

            # fit the currentLow data to the specified order
            P = np.polyfit(xRange, yRange, fitOrder)
            currentLowFit.append(np.polyval(P,timeFit[len(timeFit)-1]))

            # move the fitting center to the adjacent point in currentLow
            currentLowFit.append(currentLowAvg[i+init_pt])
            timeFit.append(timeLowAvg[i+init_pt])

            # reset the fitting variables
            xRange[:] = []
            yRange[:] = []
            del P

            # store the range of currentHigh data to be fitted
            for j in range(0, fitWindow):
                xRange.append(timeHighAvg[i+j+1])
                yRange.append(currentHighAvg[i+j+1])
            
            # fit the currentHigh data to the specified order
            P = np.polyfit(xRange, yRange, fitOrder)
            currentHighFit.append(np.polyval(P, timeFit[len(timeFit)-1]))

    a = np.array(currentLowFit)
    return timeFit, currentLowFit, currentHighFit

def extractSignal (timeFit, currentLowFit, currentHighFit):
    """ Extract the deltaI and finally the change in zeta potential, which constitutes the signal from the binding of 
    the biomolecule """
    currentLowFit = np.array(currentLowFit)
    currentHighFit = np.array(currentHighFit)
    delta_I = currentHighFit - currentLowFit

    # parameters for calculation of zeta potential, all in SI units
    visc = 0.8 * 10**-3         # viscosity of PBS buffer
    leng = 5.4 * 10**-2         # capillary length
    area = 490 * 10**-12        # inner surface area of the capillary
    perm = 681.85 * 10**-12     # permittivity of PBS buffer
    delta_P = (3-1.5) * 10**5   # pressure difference

    # zeta potential formula
    zetaPotential =10**-9*delta_I*leng*visc/(area*perm*delta_P)

    # Using Savitzky-Golay filter to remove high-frequency noise
    filtZeta = sp.savgol_filter(zetaPotential, 19, 3)
    return delta_I, filtZeta

# Data import
inputCol1 = 0
inputCol2 = 1
startRow = 1
delim = '\t'
fileName = 'input_data_with-header.txt'
timeIn, currentIn = dataImport(inputCol1, inputCol2, startRow, delim, fileName)

# Data Selection
selectRange1 = [610,1236]
selectRange2 = [1341,2441]
selectRange3 = [2483,3203]
timeSelect, currentSelect = dataSelector(selectRange1, selectRange2, selectRange3)

# Remove noise from pressure pulsing
intraPulseNoise = 0.0047#2*(0.5592-0.558)/0.558
interPulseNoise = 0.0065
timeRefine, currentRefine = noiseRemover(timeSelect, currentSelect, intraPulseNoise, interPulseNoise)

# Segregate data
startCheck = 1
thresh = (648-634)*0.3
timeLow, currentLow, timeHigh, currentHigh = dataSegregator(timeRefine, currentRefine, startCheck, thresh)

# Average over each pulse
pulseWidth = 0.5
timeLowAvg, currentLowAvg = pulseAveraging(timeLow, currentLow, pulseWidth)
timeHighAvg, currentHighAvg = pulseAveraging(timeHigh, currentHigh, pulseWidth)

# Fit the two sets of data with a selected order and fitting window to interpolate the alternate gaps
fitOrder = 3
fitWindow = 4
fitRange = len(timeLowAvg) if len(timeLowAvg) < len(timeHighAvg) else len(timeHighAvg)
timeFit, currentLowFit, currentHighFit = piecewiseNonLinearInterp(fitOrder, fitWindow, fitRange, timeLowAvg, currentLowAvg, timeHighAvg, currentHighAvg, startCheck)

# Extract the signal
delta_I, zetaPotential = extractSignal(timeFit, currentLowFit, currentHighFit)

# Plot the results
plt.figure(1)
plt.plot(timeIn, currentIn,'.-', linewidth = 0.3)
plt.title('Raw input data')
plt.ylabel('Streaming Current (pA)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('1a_input_data.png')
# plt.show()

plt.figure(2)
plt.plot(timeIn, currentIn,'o-')
plt.title('Raw input data (Zoomed version)')
plt.ylabel('Streaming Current (pA)')
plt.xlabel('Time (min)')
plt.xlim(18, 20)
plt.ylim(450, 520)
plt.grid()
plt.savefig('1b_zomed_input_data.png')
# plt.show()

plt.figure(3)
plt.plot(timeSelect, currentSelect,'.-', linewidth = 0.3)
plt.title('Selected Data')
plt.ylabel('Streaming Current (pA)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('2_selected_data')

plt.figure(4)
plt.plot(timeRefine, currentRefine,'.-', linewidth = 0.3)
plt.title('Refined Data')
plt.ylabel('Streaming Current (pA)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('3_refined_data')

plt.figure(5)
plt.plot(timeLow, currentLow,'r.', timeHigh, currentHigh, 'b.')
plt.legend(('Low', 'High'), loc='best', shadow=True, fontsize='x-large')
plt.title('Segregated Data')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('4_segregated_data')

plt.figure(6)
plt.plot(timeLowAvg, currentLowAvg,'.', timeHighAvg, currentHighAvg, '.')
plt.legend(('Low', 'High'), loc='best', shadow=True, fontsize='x-large')
plt.title('Averaged over each pulse')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('5_averaged_data')

plt.figure(7)
plt.plot(timeFit, currentLowFit,'o-', timeFit, currentHighFit, 'o-')
plt.legend(('Low', 'High'), loc='best', shadow=True, fontsize='x-large')
plt.title('Fitted piecewise')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('6_fitted_data')

plt.figure(8)
plt.plot(timeFit, zetaPotential, '-')
plt.title('Zeta Signal')
plt.ylabel('Zeta Potential (mV)')
plt.xlabel('Time (min)')
plt.grid()
plt.savefig('7_signal_data')

plt.show()

# # Profiler end
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())

# # Timer end
# print('Total time for code execution',time.time()-start_time)

