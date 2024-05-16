import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter

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

def GetBunchStructureEvolutionMatrix(ToAs, LHCOrbitTime, OrbitBins, TimeBins):
    TemporalMatrix = np.zeros(( TimeBins.shape[0]-1, OrbitBins.shape[0]-1))
    for i in range(TimeBins.shape[0]-1):
        ToASection = ToAs[(ToAs>TimeBins[i]) & (ToAs<TimeBins[i+1])].copy()
        y,x = np.histogram((ToASection)%(LHCOrbitTime),bins=OrbitBins)

        TemporalMatrix[i,:] = y
    return TemporalMatrix
    
def FindBestAllignmentShift(function1, function2):
    # Calculate the cross-correlation of the two functions

    correlation = correlate(function1, function2)

    # Find the shift that maximizes the correlation
    best_shift = np.argmax(correlation) - (len(function1) - 1)
    
    return best_shift
"""def FindBestAllignmentShift(function1, function2, max_shift):
    # Calculate the cross-correlation of the two functions

    correlation = correlate(function1, function2, mode='full')

    # Find the range of indices within the specified max_shift
    start_index = len(function1) - 1 - max_shift
    end_index = len(function1) - 1 + max_shift + 1

    # Find the shift that maximizes the correlation within the specified range
    best_shift = np.argmax(correlation[start_index:end_index]) - max_shift
    
    return best_shift"""
def find_best_alignment_fourier(signal_ref, signal_to_align, frequency_range):
    """
    Finds the best alignment shift between two signals within a specified frequency range using the Fourier transform.

    Args:
        signal_ref (numpy.ndarray): Reference signal.
        signal_to_align (numpy.ndarray): Signal to align with the reference.
        frequency_range (tuple): Frequency range of interest (start_freq, end_freq).

    Returns:
        int: Best alignment shift (in number of samples).
    """
    # Perform the Fourier transform
    ref_transform = np.fft.fft(signal_ref)
    align_transform = np.fft.fft(signal_to_align)

    # Frequency range indices
    freq_range_indices = np.arange(len(signal_ref)) * (1 / len(signal_ref))

    # Find the indices within the frequency range
    freq_indices = np.where((freq_range_indices >= frequency_range[0]) & (freq_range_indices <= frequency_range[1]))[0]

    # Apply a mask to the Fourier transforms
    ref_transform_masked = np.zeros_like(ref_transform)
    align_transform_masked = np.zeros_like(align_transform)
    ref_transform_masked[freq_indices] = ref_transform[freq_indices]
    align_transform_masked[freq_indices] = align_transform[freq_indices]

    # Calculate the cross-correlation using the inverse Fourier transform
    cross_corr = np.fft.ifft(ref_transform_masked * np.conj(align_transform_masked))

    # Find the index of the maximum correlation
    best_shift_index = np.argmax(np.abs(cross_corr))

    # Calculate the best shift in number of samples
    best_shift = best_shift_index - len(signal_ref) + 1

    return best_shift
    
def monitor_alignment_stability(shifts, threshold=0.01, num_iterations=5):
    """
    Monitor the stability of alignment based on the variation of alignment parameters (shifts) across iterations.

    Args:
        shifts (numpy.ndarray): Array of alignment shifts across iterations.
        threshold (float): Threshold for considering the alignment stable. Defaults to 0.01.
        num_iterations (int): Number of iterations to consider for stability assessment. Defaults to 5.

    Returns:
        bool: True if alignment is considered stable, False otherwise.
    """
    if len(shifts) < num_iterations:
        return False

    recent_shifts = shifts[-num_iterations:]
    max_shift_variation = np.max(np.abs(np.diff(recent_shifts)))

    return max_shift_variation < threshold
"""def low_pass_filter(signal, sampling_freq,cutoff_freq,  order=5):#
    # Discrete Fourier Transform
    frequency_domain = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_freq)

    # Filter design
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist_freq
    filter_window = np.zeros_like(signal)
    filter_window[np.abs(frequencies) <= normalized_cutoff] = 1

    # Apply filter to frequency domain
    filtered_frequency_domain = frequency_domain * filter_window

    # Inverse Fourier Transform
    filtered_signal = np.fft.ifft(filtered_frequency_domain).real

    return filtered_signal"""

def low_pass_filter(signal, sampling_freq,cutoff_freq,  order=5):
    # Discrete Fourier Transform
    b,a = butter(order, cutoff_freq, fs=sampling_freq)
    filtered_signal = lfilter(b,a,signal)

    return filtered_signal
    
def FindClockDriftShiftCorrection(ToAs, LHCOrbitTime, OrbitBins, TimeBins,MaximumShift=3, MaxIterations=30,method="Correlation"):
    CorrectedToAs = ToAs.copy()
    
    ShiftVectors = []
    Interpolations  = []
    MeanCorrelations = np.zeros(MaxIterations)
    CurrentShiftSum = 1
    IterationNumber = 0
    FilterFreq = 0.12#*(OrbitBins[1]-OrbitBins[0])
    SamplingFreq=1
    PreviousShiftVector = np.zeros(TimeBins.shape[0]-2)
    TotalShiftVector = np.zeros(TimeBins.shape[0]-1)
    
    BunchFrequency = (1.0/15.96779293)*(OrbitBins[1]-OrbitBins[0])
    while(((CurrentShiftSum)>0) & (IterationNumber<MaxIterations)):
        
        Values = GetBunchStructureEvolutionMatrix(CorrectedToAs, LHCOrbitTime, OrbitBins, TimeBins)
        #Values[0,:] = low_pass_filter(Values[0,:],SamplingFreq,FilterFreq)#
        Values[0,:] = (Values[0,:]-Values[0,:].mean())/Values[0,:].std()
        Correlations = np.zeros(TimeBins.shape[0]-1)
        for i in range(1, TimeBins.shape[0]-1):
            #Values[i,:] = low_pass_filter(Values[i,:],SamplingFreq,FilterFreq)#
            Values[i,:] = (Values[i,:]-Values[i,:].mean())/Values[i,:].std()
            Correlations[i] = np.abs(np.corrcoef(Values[i-1,:] ,Values[i,:] )[0, 1])
        MeanCorrelations[IterationNumber] = np.abs(Correlations).mean()
        y = Values.sum(axis=0)
        peaks, _ = find_peaks(y,height=30)
        CurrentShiftVector = np.zeros(TimeBins.shape[0]-1)
        PreviousRow = Values[0,:]
        #PreviousRow = low_pass_filter(PreviousRow,SamplingFreq,FilterFreq)
        PreviousRow = (PreviousRow-PreviousRow.mean())/PreviousRow.std()
        for i in range(1,TimeBins.shape[0]-1):
            CurrentRow = Values[i,:]
            #CurrentRow = low_pass_filter(CurrentRow,SamplingFreq,FilterFreq)
            CurrentRow = (CurrentRow-CurrentRow.mean())/CurrentRow.std()
            #CurrentShiftVector[i] = FindBestAllignmentShift(PreviousRow, CurrentRow,MaximumShift)
            ##,MaximumShift)
            if(method=="Correlation"):
                CurrentShiftVector[i] = FindBestAllignmentShift(PreviousRow, CurrentRow)
            elif(method=="Fourier"):
                CurrentShiftVector[i] = find_best_alignment_fourier(PreviousRow, CurrentRow, (0.0,1.0))
            else:
                print("Warning: Unknown method specified, Correlation method is being used.")
                CurrentShiftVector[i] = FindBestAllignmentShift(PreviousRow, CurrentRow, )
            #CurrentShiftVector[i] = (CurrentShiftVector[i]+(IterationNumber)*PreviousShiftVector[i])//(IterationNumber+1)
            #if(CurrentShiftVector[i] == -PreviousShiftVector[i]):
            #    CurrentShiftVector[i]=CurrentShiftVector[i]//2
            PreviousRow = CurrentRow
        PreviousShiftVector = CurrentShiftVector.copy()
        
        CurrentShiftSum = abs(CurrentShiftVector).sum()     
        #TotalShiftVector += CurrentShiftVector   
        #TotalShiftVector[TotalShiftVector>MaximumShift] = MaximumShift
        #TotalShiftVector[TotalShiftVector<-MaximumShift] = -MaximumShift
        #CurrentShiftSum = abs(TotalShiftVector).sum()
        
        
        CurrentShiftVector = np.cumsum(CurrentShiftVector)
        #CummedSummedTotalShiftVector = np.cumsum(TotalShiftVector)
        
        inter = np.interp(CorrectedToAs, TimeBins[:-1], CurrentShiftVector)*(OrbitBins[1]-OrbitBins[0]) 
        #inter = np.interp(ToAs, TimeBins[:-1], CummedSummedTotalShiftVector)*(OrbitBins[1]-OrbitBins[0])
        
        CorrectedToAs = CorrectedToAs + inter 
        #CorrectedToAs = ToAs + inter
        
        
        
        ShiftVectors.append(CurrentShiftVector)
        #ShiftVectors.append(CummedSummedTotalShiftVector)
        
        print(IterationNumber, CurrentShiftSum,MeanCorrelations[IterationNumber], peaks.shape[0])
        
        IterationNumber+=1
    return ShiftVectors

"""def ApplyToAShiftCorrection(ToA, ShiftVectors, TimeBins, OrbitBinSize):
    CorrectedToAs = ToA.copy()
    for shift in ShiftVectors:
        inter = np.interp(CorrectedToAs, TimeBins[:-1], shift)*OrbitBinSize
        CorrectedToAs += inter
    return CorrectedToAs"""
def ApplyToAShiftCorrection(ToA, ShiftVectors, TimeBins, OrbitBinSize):
    CorrectedToAs = ToA.copy()
    for shift in ShiftVectors:
        #inter = shift[np.digitize(CorrectedToAs, TimeBins[:-1])-1]*OrbitBinSize
        inter = np.interp(CorrectedToAs, TimeBins[:-1], shift)*OrbitBinSize
        
        CorrectedToAs += inter #+ Addition
        
    return CorrectedToAs
def FindShiftBetweenToALists(ListOfToAFiles,AllignVector, LHCOrbitTime, OrbitBins, MaxIterations=5):
    BunchStructures = []
    for ToAs in ListOfToAFiles:
        y,x = np.histogram((ToAs)%(LHCOrbitTime),bins=OrbitBins)
        y = y*(y<y.max()*0.5)
        BunchStructures.append(y)
    ShiftVectors = []
    CurrentShiftSum = 1
    IterationNumber = 0
    while((IterationNumber<MaxIterations)):
        
        CurrentShiftVector = np.zeros(len(ListOfToAFiles)).astype(int)
    
        for i in range(len(ListOfToAFiles)):
            CurrentRow = BunchStructures[i]
            if(CurrentRow.sum()>0):
                #RandomShift = np.random.randint(int(OrbitBins.shape[0]))
                #CurrentRow = np.roll(BunchStructures[i],RandomShift)
                CurrentRow = CurrentRow/CurrentRow.sum()
                CurrentShiftVector[i] = FindBestAllignmentShift(AllignVector, CurrentRow)# - RandomShift
                BunchStructures[i] = np.roll(BunchStructures[i],CurrentShiftVector[i])
                
        CurrentShiftSum = np.abs(CurrentShiftVector).sum()
        
        ShiftVectors.append(CurrentShiftVector)
        
        print(CurrentShiftSum, end=" ")
        IterationNumber+=1
    ToAShifts = np.zeros(len(ListOfToAFiles))
    for i in ShiftVectors:
        ToAShifts += i
    ToAShifts *= (OrbitBins[1] - OrbitBins[0])
    return ToAShifts