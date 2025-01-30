# Data Preparation Guide

## Overview
This document outlines the process for preparing waveform data for model input. It details the required input formats, filtering procedures, normalization steps, and parameter settings.

## Input Requirements
- Original waveform data must be available in `.w` format
- Associated metadata must be present in `.wfdisc` file format
- Input data can be from either 3C stations or array stations

## Processing Steps

### 1. Data Loading
- Locate the `.w` file containing the waveform data
- Find the corresponding `.wfdisc` file with station metadata
- For 3C stations: Select only the Z channel (BHZ)
- For array stations: Construct the beam (we use ARA formula).

### 2. Waveform Processing
1. Apply linear detrend correction using:
   ```python
   obspy.detrend('linear')
   ```

2. Apply filtering based on `.wfdisc` filter specifications

    ```python
   bandpass(waveform, 0.8, 4.0, 3, 'BP', True)
   ```

3. Perform interpolation to achieve 40Hz frequency

    ```python
   np.interp(
      np.arange(0, 5, 0.025), 
      np.arange(0, 5, 5/len(waveform)), 
      waveform
   )
   ```

### 3. Time Window Selection
- Identify ARRTIME (arrival time)
- Identify AMPTIME (amplitude time)
- Validate that AMPTIME falls within window: [ARRTIME, ARRTIME + 5 seconds]

### 4. Model Input Preparation
- Extract time window: [ARRTIME, ARRTIME + 5 seconds]
- Ensure output shape is (200, 1)
- The resulting processed window can be used as input for the try_model notebook

## Validation Criteria
- AMPTIME must occur within 5 seconds after ARRTIME
- Final output must maintain shape (200, 1)
- Sampling frequency must be 40Hz after interpolation