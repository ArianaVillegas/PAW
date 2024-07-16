import os
import time
import h5py
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError

import pytorch_lightning as pl

from src.metrics import AmplitudeRMSE, PeriodRMSE, MagnitudeRMSE, WindowAccuracy
from src.model import PAW, PLWrapper
from src.utils import preprocess
from src.utils import AmpPerLoss
from src.dataloader import SeismicDataset


import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')


if __name__ == '__main__':
	"""
	Parse arguments
	"""
	parser = argparse.ArgumentParser(description='Test model to predict amplitude window in remaining data')
	parser.add_argument("--data-folder", type=str, default=".",
			help="The relative path to the dataset folder")
	parser.add_argument("--data", type=str, default='dataset',
			help="The name of the preprocess dataset (Only h5 files)")
	parser.add_argument("--model-id", type=int, default=1,
			help="The unique id to attach to the model")
	parser.add_argument("--padding", type=float, default=0.5,
			help="The padding of the observed window")
	parser.add_argument("--label-type", type=str, choices=['binary', 'gaussian', 'orig'], default='binary',
			help="Type of the output label shape")
	parser.add_argument("--model", type=str, choices=['paw'], default="paw",
			help="Model name")
	parser.add_argument("--loss", type=str, choices=['mse', 'bce', 'amper'], default="bce",
			help="Loss function name")
	args = parser.parse_args()

	"""
	HyperParameters
	"""
	directory = './models'
	# Build dataset
	freq = 0.025
	padding = args.padding
	metric = 'pdf'
	scale_type = 'MinMax'
	# Config training
	batch = 32
	lr = 1e-3
	if args.loss == 'bce':
		loss = torch.nn.BCELoss()
	elif args.loss == 'mse':
		loss = torch.nn.MSELoss()
	elif args.loss == 'amper':
		loss = AmpPerLoss()

	"""
	Build Dataset
	"""
	sta_3c = ['BOSA', 'CPUP', 'DBIC', 'LBTB', 'LPAZ', 'PLCA', 'VNDA']
	sta_arr = ['ASAR','BRTR','CMAR','ILAR','KSRS','MKAR','PDAR','TXAR']
	
	dir_name = os.path.dirname(os.path.abspath('__file__'))
	with h5py.File(os.path.join(dir_name, f'{args.data_folder}/{args.data}.h5'), "r") as f:
		data = f['waveforms'][()]
		labels = f['labels'][()]
	data = np.concatenate([np.zeros((data.shape[0], round((padding)/freq), 1)), 
                            data, 
                            np.zeros((data.shape[0], round((padding)/freq), 1))], axis=1)


	"""
	Configure Model
	"""
	config = {
        # Config dataset
        'padding': padding,
		'metric': 'pdf',
		'transformation': scale_type,
		'shape': data.shape,
		'out_shape': data.shape[-1],
		'lr': lr
	}
	model_ = PAW(config)
	model = PLWrapper(args, config, model_, loss, model_.__class__)
	trial_name = 'wdif_{}_{}_{}'.format(
        args.model, args.model_id, loss.__class__.__name__
    )
		
	# Load trained models
	try:
		checkpoint = torch.load(f'models/{trial_name}.pth')
		model.load_state_dict(checkpoint)
	except Exception as e:
		print(f"Model checkpoint not found. ({e})")
		exit()


	"""
	Test datasets
	"""
	# Setup trainer
	trainer = pl.Trainer(devices='auto', 
						accelerator='auto',
						benchmark=True,
						log_every_n_steps=10,
						enable_progress_bar=True)
	
	# Create dataloader
	forward_time = []
	t = np.arange(0, 5+freq, freq)
	pre_data, pre_labels = preprocess(data, labels, metric, t, freq, scale_type, padding, args.label_type)
	dataset = SeismicDataset(pre_data, pre_labels)
	loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2)

	start_time_test = time.time()
	preds = trainer.predict(model, loader)
	end_time_test = time.time()
	forward_time = (end_time_test-start_time_test)/len(loader.dataset)

	test_amp_rmse = AmplitudeRMSE(squared=False)
	test_per_rmse = PeriodRMSE(squared=False)
	test_mag_rmse = MagnitudeRMSE(squared=False)
	test_wdw_rmse = MeanSquaredError(squared=False)
	test_wdw_acc = WindowAccuracy()
	# Preds: x, y, y_hat, y_wdw, y_wdw_hat
	for i, pred in enumerate(preds):
		test_amp_rmse(pre_data[i*batch:(i+1)*batch], pred['y_wdw_hat'], pred['y_wdw'])
		test_per_rmse(pred['y_wdw_hat'], pred['y_wdw'])
		test_mag_rmse(data, pred['y_wdw_hat'], pred['y_wdw'])
		test_wdw_rmse(pred['y_hat'], pred['y'])
		test_wdw_acc(pred['y_wdw_hat'], pred['y_wdw'])
	print(f'\t Forward Time: {forward_time*1000} ms')
	print(f'\t Amplitude RMSE: {test_amp_rmse.compute()}')
	print(f'\t Period RMSE: {test_per_rmse.compute()}')
	print(f'\t Magnitude RMSE: {test_mag_rmse.compute()}')
	print(f'\t Window RMSE: {test_wdw_rmse.compute()}')
	print(f'\t Window Accuracy: {test_wdw_acc.compute()}')
