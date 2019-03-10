import os
os.system("cls")
import numpy as np
import matplotlib.pyplot as plt

# # Profiler start
# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
# pr.enable()

# # Timer start
# import time
# start_time = time.time()

def dataImport(inputCol1, inputCol2, delim, fileName):
    timeIn=[] 
    currentIn = []
    data = np.genfromtxt(fileName, delimiter=delim)
    data = np.array(data)
    timeIn = data[:,][:,inputCol1]/60.0
    currentIn = data[:,][:,inputCol2]*10**12
    timeIn = np.array(timeIn, dtype=np.float64)
    currentIn = np.array(currentIn, dtype=np.float64)
    return (timeIn, currentIn)

def dataSelector(selectRange1, selectRange2, selectRange3):
    timeSelect = np.concatenate([timeIn[selectRange1[0]:selectRange1[1]:], timeIn[selectRange2[0]:selectRange2[1]:], timeIn[selectRange3[0]:selectRange3[1]:]])
    currentSelect = np.concatenate([currentIn[selectRange1[0]:selectRange1[1]:], currentIn[selectRange2[0]:selectRange2[1]:], currentIn[selectRange3[0]:selectRange3[1]:]])
    return timeSelect, currentSelect

def noiseRemover(timeSelect, currentSelect, intraPulseNoise, interPulseNoise):
    timeRefine = []
    currentRefine = []
    for i in range(1,len(timeSelect)-1):
        relativeForwardChange = (currentSelect[i]-currentSelect[i-1])/currentSelect[i]
        relativeBackwardChange = (currentSelect[i]-currentSelect[i+1])/currentSelect[i]
        if relativeForwardChange*relativeBackwardChange < intraPulseNoise**2 and relativeForwardChange*relativeBackwardChange > -1*interPulseNoise**2:
            timeRefine.append(timeSelect[i])
            currentRefine.append(currentSelect[i])
    timeRefine = np.array(timeRefine)
    currentRefine = np.array(currentRefine)
    return timeRefine, currentRefine

def dataSegregator(timeRefine, currentRefine, switch_s, thresh):
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
        if (currentRefine[i+1]-currentRefine[i]) > thresh:
            switch_s=1
        elif (currentRefine[i+1]-currentRefine[i]) < (-1*thresh):
            switch_s=0
    return timeLow, currentLow, timeHigh, currentHigh

def pulseAveraging(time, current, pulseWidth):
    counter=0.0
    timePulseSum = 0.0
    currentPulseSum = 0.0
    timeAvg = []
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

    timeFit = []
    currentLowFit = []
    currentHighFit = []
    fitStart=1
    xRange = []
    yRange = []
    P = []

    if startCheck == 0:
        for i in range(fitStart, fitRange-fitWindow):
            timeFit.append(timeLowAvg[i+2])
            currentLowFit.append(currentLowAvg[i+2]) # do something about this 2
            xRange[:] = []
            yRange[:] = []
            P[:] = []
            for j in range(0, fitWindow):
                xRange.append(timeHighAvg[i+j])
                yRange.append(currentHighAvg[i+j])
            
            P = np.polyfit(xRange, yRange, fitOrder)
            currentHighFit.append(np.polyval(P,timeFit[i-fitStart]))
            currentHighFit.append(currentHighAvg(i+2))
            timeFit.append(timeHighAvg(i+2))
            xRange[:] = []
            yRange[:] = []
            P[:] = []
            for j in range(0, fitWindow):
                xRange.append(timeLowAvg(i+j))
                yRange.append(currentLowAvg(i+j))
    
            P = np.polyfit(xRange, yRange, fitOrder)
            currentLowFit.append(np.polyval(P, timeFit[i-fitStart]))
    
    else:
        for i in range(fitStart, fitRange-fitWindow):
            timeFit.append(timeHighAvg[i+2])
            currentHighFit.append(currentHighAvg[i+2])
            xRange[:] = []
            yRange[:] = []
            del P
            for j in range(0, fitWindow):
                xRange.append(timeLowAvg[i+j])
                yRange.append(currentLowAvg[i+j])

            P = np.polyfit(xRange, yRange, fitOrder)
            print(P)
            print(yRange)
            print(np.polyval(P,timeFit[i-fitStart]))
            currentLowFit.append(np.polyval(P,timeFit[i-fitStart]))
            currentLowFit.append(currentLowAvg[i+2])
            timeFit.append(timeLowAvg[i+2])
            xRange[:] = []
            yRange[:] = []
            del P
            for j in range(0, fitWindow):
                xRange.append(timeHighAvg[i+j])
                yRange.append(currentHighAvg[i+j])
            
            P = np.polyfit(xRange, yRange, fitOrder)
            currentHighFit.append(np.polyval(P, timeFit[i-fitStart]))
    a = np.array(currentLowFit)
    return timeFit, currentLowFit, currentHighFit

# Data import
inputCol1 = 0
inputCol2 = 1
delim = '\t'
fileName = 'input_data.txt'
timeIn, currentIn = dataImport(inputCol1, inputCol2, delim, fileName)

# Data Selection
selectRange1 = [0,1236]
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
# print(timeFit)
print(len(timeFit))
print(len(currentLowFit))
print(len(currentHighFit))

# Plot the results
plt.figure(1)
plt.plot(timeIn, currentIn,'o-')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.show()

plt.figure(2)
plt.plot(timeSelect, currentSelect,'.')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.show()

plt.figure(3)
plt.plot(timeRefine, currentRefine,'.')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.show()

plt.figure(4)
plt.plot(timeLow, currentLow,'r', timeHigh, currentHigh, 'b')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.show()

plt.figure(5)
plt.plot(timeLowAvg, currentLowAvg,'.', timeHighAvg, currentHighAvg, '.')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
plt.show()

plt.figure(6)
plt.plot(timeFit, currentLowFit,'o-')#, timeFit, currentHighFit, 'o-')
plt.ylabel('Current (pA)')
plt.xlabel('Time (min)')
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
