import h5py

with h5py.File('dataset.h5', "r") as f:
	waveforms = f['waveforms'][()]
	labels = f['labels'][()]
	
assert(waveforms.shape[0] == labels.shape[0])
print(f"Same number of waveforms ({waveforms.shape}) and window labels ({labels.shape})")
print(waveforms[0].flatten())
print(labels[0])