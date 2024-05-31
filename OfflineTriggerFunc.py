import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
from tqdm import tqdm

"""
When To
"""
def RemoveNoisyToAValues(InputDf):
    df = InputDf.copy()
    df["ToABin"] = np.digitize(df["MinToA"]%25, bins=np.linspace(-0.1,25,1000+1))*25/1000

    NoisyValues = np.array([-1])
    while(NoisyValues.size!=0):
        t = df.groupby("ToABin").size()
        OutlierPeaks, _ = find_peaks(np.append(t.values[2],t.values), threshold = t.median())
        NoisyValues = t.index[OutlierPeaks-1]
        for i in NoisyValues:
            df = df[df["ToABin"]!=i]
    return df 

"""
Takes as input:
    ToAs: This are the recorded minium ToA values of a the set of clusters in the experiment
    LHCOrbitTime: This is time it takes for a given bumch to undergo one full orbit of LHC
    OrbitBins: 
    TimeBins:
Outputs the Bunch structure evolution matrix where each row represents a all ToAs that full into the particular TimeBins
The row values is the Section(ToA) Modulus LHCOrbitTime which gives the bunch structure of the at that time

Called the evolution matrix as the bunch structure varies in due to variation in TPX3 clock bin size variations
"""
def GetBunchStructureEvolutionMatrix(ToAs, LHCOrbitTime, OrbitBins, TimeBins):
    TemporalMatrix = np.zeros(( TimeBins.shape[0]-1, OrbitBins.shape[0]-1))
    for i in range(TimeBins.shape[0]-1):
        ToASection = ToAs[(ToAs>TimeBins[i]) & (ToAs<TimeBins[i+1])].copy()
        y,x = np.histogram((ToASection)%(LHCOrbitTime),bins=OrbitBins)

        TemporalMatrix[i,:] = y
    return TemporalMatrix

"""
Takes in to vectors function1 and function2 

Calculate the optimal shift of function2 such that the cross-correlation between the two functions is optimised 

The shift is given in number of indexs to shift and can be applied using np.roll function
"""  
def FindBestAllignmentShift(function1, function2, MaximumShift=None):
    # Calculate the cross-correlation of the two functions
    correlation = correlate(function1, function2)
    
    if(MaximumShift==None):
        # Find the shift that maximizes the correlation
        best_shift = np.argmax(correlation) - (len(function1) - 1)
    else:
        # Find the range of indices within the specified MaximumShift
        start_index = len(function1) - 1 - MaximumShift
        end_index = len(function1) - 1 + MaximumShift + 1
        # Find the shift that maximizes the correlation within the specified range
        best_shift = np.argmax(correlation[start_index:end_index]) - MaximumShift
    
    return best_shift

"""
Takes in:
     ToAs: This are the recorded minium ToA values of a the set of clusters in the experiment
     LHCOribtTimeVector: This is the initial prediction of the measured relative to the TPX3 clock for each TimeBin
     Delta: This gives the range around the initial guess in which the LHC orbit time is searched per TotalIteraction, the search region:LHCOrbitTime+\- LHCOrbitTime*Delta
     OrbitBins: This the binin g
     TimeBins: The time bins in which the ToA values are split into and should have length len(LHCOribtTimeVector)+1
                    for each bin TimeBins[i]:TimeBins[i+1], LHCOribtTimeVector[i] is applied
     MaximumShift: This is the maximum 
     TotalIterations: This gives the number of times you the algorithm starts at the first row of the vector and iterates through the matrix
     SubIterations: This gives the number of times the algorithm 
     method:

The algorithm implements a hybrid version of the overshoot-undershoot algorithm to find the optimal value of LHCOrbitTimeVector
    1. Calculates the LH
"""
def FindClockDriftShiftCorrection(ToAs, LHCOrbitTimeVector, Delta, OrbitBins, TimeBins,MaximumShift=3, TotalIterations=2,SubIterations=30,method="forward", MaxShift=-1):
    #Set 
    if(MaxShift==-1):
        MaxShift=OrbitBins.shape[0]//1000
    
    for IterationNumber in range(TotalIterations):
        ToASection = ToAs[(ToAs>TimeBins[0]) & (ToAs<TimeBins[1])].copy()
        t = TimeBins[1]
        t = t%LHCOrbitTimeVector[0]
        PreviousRow,x = np.histogram((ToASection)%(LHCOrbitTimeVector[0]),bins=OrbitBins)
        #PreviousRow = (PreviousRow-PreviousRow.mean())/PreviousRow.std()
        #PreviousRow = (PreviousRow-PreviousRow.mean())
        PreviousRow = PreviousRow/np.sum(PreviousRow)
        AverageNumberOfIterationNeeded=0
        AverageCrossCorelation=0
        print(f"%d/%d"%(IterationNumber+1,TotalIterations), end=":  ")
        for i in tqdm(range(1,TimeBins.shape[0]-1)):
            if(method=="forward"):
                LHCOrbitTimeVectorMax = LHCOrbitTimeVector[i-1] + Delta*LHCOrbitTimeVector[i-1]
                LHCOrbitTimeVectorMin = LHCOrbitTimeVector[i-1] - Delta*LHCOrbitTimeVector[i-1]
                LHCOrbitTimeVector[i] = LHCOrbitTimeVector[i-1]
            elif(method=="central"):
                LHCOrbitTimeVectorMax = LHCOrbitTimeVector[i] + Delta*LHCOrbitTimeVector[i]
                LHCOrbitTimeVectorMin = LHCOrbitTimeVector[i] - Delta*LHCOrbitTimeVector[i]
            else:
                raise Exception(f"Unknown method given \"{method}\" expected \"forward\" or \"central\"")
            ToASection = ToAs[(ToAs>TimeBins[i]) & (ToAs<TimeBins[i+1])].copy()
            ToASection = ToASection - TimeBins[i]
            ToASection = ToASection + t
            #print()
            for SubIterationNumber in range(SubIterations):
                AverageNumberOfIterationNeeded+=1
                CurrentRow,x = np.histogram((ToASection)%(LHCOrbitTimeVector[i]),bins=OrbitBins)
                #CurrentRow = (CurrentRow-CurrentRow.mean())/CurrentRow.std()
                #CurrentRow = (CurrentRow-CurrentRow.mean())
                CurrentRow = CurrentRow/np.sum(CurrentRow)
                #print(SubIterationNumber,LHCOrbitTimeVector[i], end=" ")
                shift = FindBestAllignmentShift(PreviousRow, CurrentRow,MaxShift)
                if(shift<0):
                    LHCOrbitTimeVectorMin = LHCOrbitTimeVector[i]
                    LHCOrbitTimeVector[i] = (LHCOrbitTimeVectorMax+LHCOrbitTimeVectorMin)/2
                elif(shift>0):
                    LHCOrbitTimeVectorMax = LHCOrbitTimeVector[i]
                    LHCOrbitTimeVector[i] = (LHCOrbitTimeVectorMax+LHCOrbitTimeVectorMin)/2
                else:
                    LHCOrbitTimeVectorMax = LHCOrbitTimeVector[i] 
                    LHCOrbitTimeVectorMin = LHCOrbitTimeVector[i] 
                    break
            CurrentRow,x = np.histogram((ToASection)%(LHCOrbitTimeVector[i]),bins=OrbitBins)
            #CurrentRow = (CurrentRow-CurrentRow.mean())/CurrentRow.std()
            CurrentRow = CurrentRow/np.sum(CurrentRow)
            #print(AverageCrossCorelation, np.dot(PreviousRow,CurrentRow))
            AverageCrossCorelation += np.dot(PreviousRow,CurrentRow)
            t = t + (TimeBins[i+1] - TimeBins[i])
            t = t%LHCOrbitTimeVector[i]
            PreviousRow = CurrentRow
        
        AverageNumberOfIterationNeeded = AverageNumberOfIterationNeeded/(TimeBins.shape[0]-2)
        AverageCrossCorelation = AverageCrossCorelation/(TimeBins.shape[0]-2)
        print(f"Average Number Of iterations needed: %.5g, Average Zero-Lag Cross-Correlation: %.4e"%(AverageNumberOfIterationNeeded,AverageCrossCorelation))
        if(method=="forward"):
            LHCOrbitTimeVectorMax = LHCOrbitTimeVector[1] + Delta*LHCOrbitTimeVector[1]
            LHCOrbitTimeVectorMin = LHCOrbitTimeVector[1] - Delta*LHCOrbitTimeVector[1]
            LHCOrbitTimeVector[0] = LHCOrbitTimeVector[1]
        elif(method=="central"):
            LHCOrbitTimeVectorMax = LHCOrbitTimeVector[0] + Delta*LHCOrbitTimeVector[0]
            LHCOrbitTimeVectorMin = LHCOrbitTimeVector[0] - Delta*LHCOrbitTimeVector[0]
            
        else:
                raise Exception(f"Unknown method given \"{method}\" expected \"forward\" or \"central\"")
                
        t1 = ToAs[(ToAs>TimeBins[0]) & (ToAs<TimeBins[1])].copy()
        t2 = ToAs[(ToAs>TimeBins[1]) & (ToAs<TimeBins[2])].copy()
        for SubIterationNumber in range(SubIterations):
            tt2 = t2 - TimeBins[1] + (TimeBins[1]%LHCOrbitTimeVector[0])
            PreviousRow,x = np.histogram((tt2)%(LHCOrbitTimeVector[1]),bins=OrbitBins)
            #PreviousRow = (PreviousRow-PreviousRow.mean())/PreviousRow.std()
            #PreviousRow = (PreviousRow-PreviousRow.mean())
            PreviousRow = PreviousRow/np.sum(PreviousRow)
            CurrentRow,x = np.histogram((t1)%(LHCOrbitTimeVector[0]),bins=OrbitBins)
            #CurrentRow = (CurrentRow-CurrentRow.mean())/CurrentRow.std()
            CurrentRow = CurrentRow/np.sum(CurrentRow)
            shift = FindBestAllignmentShift(PreviousRow, CurrentRow,MaxShift)
            if(shift<0):
                LHCOrbitTimeVectorMin = LHCOrbitTimeVector[0]
                LHCOrbitTimeVector[0] = (LHCOrbitTimeVectorMax+LHCOrbitTimeVectorMin)/2
            elif(shift>0):
                LHCOrbitTimeVectorMax = LHCOrbitTimeVector[0]
                LHCOrbitTimeVector[0] = (LHCOrbitTimeVectorMax+LHCOrbitTimeVectorMin)/2
            else:
                LHCOrbitTimeVectorMax = LHCOrbitTimeVector[0] 
                LHCOrbitTimeVectorMin = LHCOrbitTimeVector[0] 
                break
    return LHCOrbitTimeVector

"""

"""
def GetBunchStructureEvolutionMatrix(ToAs, LHCOrbitTimeVector, OrbitBins, TimeBins):
    TemporalMatrix = np.zeros(( TimeBins.shape[0]-1, OrbitBins.shape[0]-1))
    tt= np.int64(0)
    for i in range(TimeBins.shape[0]-1):
        ToASection = ToAs[(ToAs>TimeBins[i]) & (ToAs<TimeBins[i+1])].copy()
        ToASection -= TimeBins[i]
        ToASection += tt
        y,x = np.histogram((ToASection)%(LHCOrbitTimeVector[i]),bins=OrbitBins)
        TemporalMatrix[i,:] = y
        if(i!=TimeBins.shape[0]-2):
            tt += (TimeBins[i+1] - TimeBins[i])
            tt = tt%LHCOrbitTimeVector[i]
    return TemporalMatrix
def GetBunchStructureEvolutionMatrixWithVariableOrbitTime(ToAs, LHCOrbitTimeVector, OrbitBinNumber, TimeBins):
    TemporalMatrix = np.zeros(( TimeBins.shape[0]-1, OrbitBinNumber-1))
    tt= np.int64(0)
    for i in range(TimeBins.shape[0]-1):
        ToASection = ToAs[(ToAs>TimeBins[i]) & (ToAs<TimeBins[i+1])].copy()
        ToASection -= TimeBins[i]
        ToASection += tt
        y,x = np.histogram((ToASection)%(LHCOrbitTimeVector[i]),bins=np.linspace(0,LHCOrbitTimeVector[i],OrbitBinNumber))
        TemporalMatrix[i,:] = y
        if(i!=TimeBins.shape[0]-2):
            tt += (TimeBins[i+1] - TimeBins[i])
            tt = tt%LHCOrbitTimeVector[i]
    return TemporalMatrix   

"""

"""
def FindClockDriftShiftCorrectionByInterpolation(ToAs, LHCOrbitTime, OrbitBins, TimeBins,MaximumShift=3, MaxIterations=30,method="Correlation"):
    CorrectedToAs = ToAs.copy()
    
    ShiftVectors = []
    MeanCorrelations = np.zeros(MaxIterations)
    CurrentShiftSum = 1
    PreviousShiftVector = np.zeros(TimeBins.shape[0]-2)
    
    for IterationNumber in range(MaxIterations):
        Values = GetBunchStructureEvolutionMatrix(CorrectedToAs, LHCOrbitTime, OrbitBins, TimeBins)
        Values[0,:] = (Values[0,:]-Values[0,:].mean())/Values[0,:].std()
        Correlations = np.zeros(TimeBins.shape[0]-1)
        for i in range(1, TimeBins.shape[0]-1):
            Values[i,:] = (Values[i,:]-Values[i,:].mean())/Values[i,:].std()
            Correlations[i] = np.abs(np.corrcoef(Values[i-1,:] ,Values[i,:] )[0, 1])
        MeanCorrelations[IterationNumber] = np.abs(Correlations).mean()
        y = Values.sum(axis=0)
        peaks, _ = find_peaks(y,height=30)
        CurrentShiftVector = np.zeros(TimeBins.shape[0]-1).astype('int64')
        PreviousRow = Values[0,:]
        PreviousRow = (PreviousRow-PreviousRow.mean())/PreviousRow.std()
        for i in range(1,TimeBins.shape[0]-1):
            CurrentRow = Values[i,:]
            CurrentRow = (CurrentRow-CurrentRow.mean())/CurrentRow.std()
            CurrentShiftVector[i] = FindBestAllignmentShift(PreviousRow, CurrentRow)
            PreviousRow = CurrentRow
        PreviousShiftVector = CurrentShiftVector.copy()
        
        CurrentShiftSum = abs(CurrentShiftVector).sum()     
        
        
        CurrentShiftVector = np.cumsum(CurrentShiftVector)
        
        inter = np.interp(CorrectedToAs, TimeBins[:-1], CurrentShiftVector)*(OrbitBins[1]-OrbitBins[0]) 
        
        CorrectedToAs = CorrectedToAs + inter 
        
        ShiftVectors.append(CurrentShiftVector)
        
        print(IterationNumber, CurrentShiftSum,MeanCorrelations[IterationNumber], peaks.shape[0])
        
    return ShiftVectors
"""

""" 
def ApplyToAShiftCorrection(ToA, ShiftVectors, TimeBins, OrbitBinSize):
    CorrectedToAs = ToA.copy()
    for shift in ShiftVectors:
        inter = np.interp(CorrectedToAs, TimeBins[:-1], shift)*OrbitBinSize
        CorrectedToAs += inter
    return CorrectedToAs
