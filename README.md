# PAW: Seismic Phase Amplitude and Period Waveform Detection

## Abstract

Subsurface earthquakes and explosions generate seismic wavefields that are recorded as time-domain signals on sensor networks around the world. To compute key characteristics such as the magnitude of these seismic events, analysts must detect and select the cleanest indicators of seismic phase amplitudes and periods in noisy signals. 

Existing automated systems designed to pick seismic phase amplitudes and periods require frequent adjustments by human analysts, which becomes a nuisance when the volume of data to process grows large. To address this problem, we have developed a neural network model that accurately replicates the performance of a human analyst 80% of the time and shows potential for decreasing analyst burden significantly. We have performed multiple tests on the model and report on its performance compared to existing deep learning techniques. 

## Dataset Details

Download dataset from [here](https://huggingface.co/datasets/suroRitch/PAW/tree/main)

The dataset consists of 80,648 waveforms. The `dataset.h5` file contains two keys:
- `waveforms`: A numpy array with a shape of (80648, 200, 1) representing the waveforms.
- `labels`: A numpy array with a shape of (80648, 2) containing labels (start time, end time relative to a 5 second window) associated with the waveforms.

## Testing Instructions

To validate our results, please follow these steps:

1. **Create Conda Environment**: 
  Use the provided `environment.yml` file to create a Conda environment with all necessary dependencies.
   ```bash
    conda env create -f environment.yml
    conda activate paw
   ```

1. **Read Dataset**:
   Utilize the `read_dataset.py` script to read the dataset required for testing.
   ```bash
    python read_dataset.py
   ```

2. **Testing Model**:
   Run the `test.py` script in two different setups:
   - Using Binary Cross-Entropy (BCE) loss
        ```bash
        python test_subset.py --loss bce
        ```
   - Using Amper loss
        ```bash
        python test_subset.py --loss amper
        ```