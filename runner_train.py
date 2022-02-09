import argparse
import datetime
import json
import os
import platform
import sys
import math
import time
import pickle
from collections import OrderedDict

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from tqdm import tqdm

from src.data.pdbbind_utils import *
from src.model.network import *


def limitNumberSize(num):
	if num > 100000:
		limited_number = "{:.3e}".format(num)
	else:
		limited_number = str(num)
	return limited_number

# no RNN
#train and evaluate
def train_and_eval(train_data, valid_data, test_data, params, paramsExt, batch_size=32, num_epoch=30,
				   results_dir_path='', rep_fold=None):
	if rep_fold is None:
		rep_fold = [0, 0]
	init_A, init_B, init_W = loading_emb(measure)
	net = APIP(init_A, init_B, init_W, params, paramsExt)
	model_path =  os.path.join(results_dir_path, f'model_rep{rep_fold[0]}_fold{rep_fold[1]}.pt')
	net.cuda()
	net.apply(weights_init)
	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print('total num params', pytorch_total_params)

	# criterion1 = nn.MSELoss()
	# criterion2 = Masked_BCELoss()
	if args.pair_loss == 'weighted_bce':
		criterion2 = Weighted_Masked_BCELoss()

	else:
		criterion2 = Masked_BCELoss()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0, amsgrad=True)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

	min_rmse=0
	#max_auc = 0
	for epoch in range(num_epoch):
		train_output_list = []
		train_label_list = []
		total_loss = 0
		# affinity_loss = 0
		pairwise_loss = 0
		total_sample = 0

		net.train()
		with tqdm(train_data, unit="batch") as tepoch:

			with tqdm(total=len(tepoch), bar_format="{postfix}") as line2:

					for i, (
					vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, affinity_label, pairwise_mask,
					pairwise_label) in enumerate(tepoch):
						tepoch.set_description(f"Epoch {epoch}")

						optimizer.zero_grad()
						pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)

						# loss_aff = criterion1(affinity_pred, affinity_label)
						loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
						loss = loss_pairwise

						total_loss += float(loss.data*batch_size)
						# affinity_loss += float(loss_aff.data*batch_size)
						pairwise_loss += float(loss_pairwise.data*batch_size)

						loss.backward()
						nn.utils.clip_grad_norm_(net.parameters(), 5)
						optimizer.step()

						divider = float(i+1)


						postfix = OrderedDict({
							'loss': limitNumberSize(round(total_loss / divider, 4)),
						    # 'affinity loss':  limitNumberSize(round(affinity_loss / divider, 4)),
							'pairwise loss': limitNumberSize(round(pairwise_loss / divider, 4)),
							'lr': optimizer.param_groups[0]['lr']
						})

						line2.set_postfix(postfix,
										  refresh=False
										  )
						line2.update()
		scheduler.step()

		net.eval()

		# loss_list = [total_loss, affinity_loss, pairwise_loss]
		# loss_name = ['total loss', 'affinity loss', 'pairwise loss']
		# print_loss = [loss_name[i]+' '+str(round(loss_list[i]/float(len(train_data)), 6)) for i in range(len(loss_name))]
		# print('epoch:', epoch, ' '.join(print_loss))

		perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']
		if epoch % 10 == 0:
			train_performance = test(net, train_data, batch_size)
			print_perf = [perf_name[i]+' '+str(round(train_performance[i], 6)) for i in range(len(perf_name))]
			print('train',  ' '.join(print_perf))

		valid_performance = test(net, valid_data, batch_size)
		print_perf = [perf_name[i]+' '+str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
		print('valid',  ' '.join(print_perf))

		if valid_performance[3] > min_rmse:
		#if valid_performance[-1] > max_auc:
			min_rmse = valid_performance[3]
			torch.save(net, model_path)
			#max_auc = valid_performance[-1]
			test_performance = test(net, test_data, batch_size)
		print_perf = [perf_name[i]+' '+str(round(test_performance[i], 6)) for i in range(len(perf_name))]
		print('test ',  ' '.join(print_perf))


	print('Finished Training')
	return test_performance


@torch.no_grad()
def test(net, test_data, batch_size):
	# output_list = []
	output_gpu = torch.FloatTensor([]).cuda()
	# label_list = []
	label_gpu = torch.FloatTensor([]).cuda()
	pairwise_auc_list = []

	for i, (vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, affinity_label, pairwise_mask,
			pairwise_label) in enumerate(test_data):

		pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)

		for j in range(len(pairwise_mask)):
			if pairwise_mask[j]:
				num_vertex = int(torch.sum(vertex_mask[j,:]))
				num_residue = int(torch.sum(seq_mask[j,:]))
				pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
				pairwise_label_i = pairwise_label[j, :num_vertex, :num_residue].reshape(-1).cpu()
				pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))

	average_pairwise_auc = np.mean(pairwise_auc_list)
	
	test_performance = [0, 0, 0, average_pairwise_auc]
	return test_performance

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run arguments.')
	parser.add_argument('measure', choices=['IC50', 'KIKD', 'All'])
	parser.add_argument('setting', choices=['new_compound', 'new_protein', 'new_new'])
	parser.add_argument('clu_thre', choices=['0.3', '0.4', '0.5','0.6'])
	parser.add_argument('--maxlen', dest='max_sequence_length', type=int, default=3072,
						help='Maximum protein sequence length')
	parser.add_argument('--cnn_kernel', dest='cnn_kernel', type=int, default=7,help='CNN kernel size')
	parser.add_argument('--transformer_depth', dest='transformer_depth', type=int, default=0,
						help='Transformer depth')
	parser.add_argument('--cnn_depth', dest='cnn_depth', type=int, default=2,
						help='Transformer depth')
	parser.add_argument('--cnn_dilation', dest='cnn_dilation', type=int, default=1,
						help='Dilation rate for cnn')
	parser.add_argument('--lambda', dest='_lambda', default='', choices=['', 'transformers', 'cnn', 'average'])
	parser.add_argument('--transformer_hidden', dest='transformer_hidden', type=int, default=128)
	parser.add_argument('--transformer_dropout', dest='transformer_dropout', type=float, default=0.1)
	parser.add_argument('--identity', dest='identity', default='', choices=['', 'cnn'])
	parser.add_argument('--pair_loss', dest='pair_loss', default='bce', choices=['bce', 'weighted_bce'])
	parser.add_argument("--lambdaValue", type=float, default=None)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--name", default='run')
	# parser.add_argument("--lambdaValue", action="store_true")


	args = parser.parse_args()
	measure = args.measure
	setting = args.setting
	clu_thre = float(args.clu_thre)

	markdown = ''
	for key in args.__dict__:
		markdown += f'{key} = {args.__dict__[key]} \n'

	result_path = '../results/'
	run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	folder_name = '_'.join([args.name, measure, setting, args.clu_thre, run_time])
	result_dir_path = os.path.join(result_path, folder_name)
	result_numpy_file = os.path.join(result_dir_path, 'result')
	result_params_file = os.path.join(result_dir_path, 'params.json')
	print(result_path)


	os.makedirs(result_dir_path, exist_ok=True)
	with open(result_params_file, 'w') as f:
		json.dump(args.__dict__, f, indent=4)


	experiment_name = ',  '.join(
		[
			str(str(param[0]) + ' = ' + str(param[1]))
			for param in sorted(args.__dict__.items())
		]
	)

	print(experiment_name)
	#evaluate scheme
	# measure = sys.argv[1]  # IC50 or KIKD
	# setting = sys.argv[2]   # new_compound, new_protein or new_new
	# clu_thre = float(sys.argv[3])  # 0.3, 0.4, 0.5 or 0.6
	n_epoch = 30
	n_rep = 5
	
	assert setting in ['new_compound', 'new_protein', 'new_new']
	assert clu_thre in [0.3, 0.4, 0.5, 0.6]
	assert measure in ['IC50', 'KIKD', 'All']
	GNN_depth, inner_CNN_depth, transformer_depth, DMA_depth = 4, args.cnn_depth, args.transformer_depth, 2
	if setting == 'new_compound':
		n_fold = 5
		k_head, transformer_head, kernel_size, hidden_size1, hidden_size2, transformer_hidden = 2, 4, args.cnn_kernel, 128, 128, args.transformer_hidden
	elif setting == 'new_protein':
		n_fold = 5
		k_head, transformer_head, kernel_size, hidden_size1, hidden_size2, transformer_hidden = 1, 2, 5, 128, 128, args.transformer_hidden
	elif setting == 'new_new':
		n_fold = 9
		# n_fold = 5
		k_head, transformer_head, kernel_size, hidden_size1, hidden_size2, transformer_hidden = 1,2, 7, 128, 128, args.transformer_hidden

	batch_size = args.batch_size
	para_names = ['GNN_depth', 'inner_CNN_depth', 'transformer_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2', 'transformer_hidden']

	params = [GNN_depth, inner_CNN_depth, transformer_depth, DMA_depth, k_head, transformer_head, kernel_size, hidden_size1, hidden_size2, transformer_hidden]
	#params = sys.argv[4].split(',')
	#params = map(int, params)

	#print evaluation scheme
	print('Dataset: PDBbind v2018 with measurement', measure)
	print('Clustering threshold:', clu_thre)
	print('Number of epochs:', n_epoch)
	print('Number of repeats:', n_rep)
	print('Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)])

	num_workers = 0
	rep_all_list = []
	rep_avg_list = []

	ind_rep_all_list = []
	ind_rep_avg_list = []
	for a_rep in range(n_rep):
		#load data
		data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold, max_sequence_length=args.max_sequence_length)


		fold_score_list = []
		ind_fold_score_list = []
		
		for a_fold in range(n_fold):
			print('repeat', a_rep + 1, 'fold', a_fold + 1, 'begin')
			train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
			print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

			train_data = data_from_index(data_pack, train_idx)
			train_data = ProteinDataset(train_data)
			train_data = torch.utils.data.DataLoader(train_data,
													 collate_fn=batch_data_process,
													 batch_size=batch_size,
													 num_workers=num_workers,
													 pin_memory=False,
													 shuffle=True)

			valid_data = data_from_index(data_pack, valid_idx)
			valid_data = ProteinDataset(valid_data)
			valid_data = torch.utils.data.DataLoader(valid_data,
													 collate_fn=batch_data_process,
													 batch_size=batch_size,
													 num_workers=num_workers,
													 pin_memory=False,
													 shuffle=True)

			test_data = data_from_index(data_pack, test_idx)
			test_data = ProteinDataset(test_data)
			test_data = torch.utils.data.DataLoader(test_data,
													collate_fn=batch_data_process,
													batch_size=batch_size,
													num_workers=num_workers,
													pin_memory=False,
													shuffle=True)

			test_performance = train_and_eval(train_data, valid_data, test_data, params, args, batch_size, n_epoch, result_dir_path, [a_rep, a_fold])

			rep_all_list.append(test_performance)
			fold_score_list.append(test_performance)
			print('-' * 30)
		print('fold avg performance', np.mean(fold_score_list, axis=0))
		rep_avg_list.append(np.mean(fold_score_list,axis=0))
		np.save(result_numpy_file, rep_all_list)

	print('all repetitions done')
	print('print all stats: RMSE, Pearson, Spearman, avg pairwise AUC')
	print('mean', np.mean(rep_all_list, axis=0))
	print('std', np.std(rep_all_list, axis=0))
	print('==============')
	print('print avg stats:  RMSE, Pearson, Spearman, avg pairwise AUC')
	print('mean', np.mean(rep_avg_list, axis=0))
	print('std', np.std(rep_avg_list, axis=0))
	print('Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)])
	#np.save('CPI_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre)+'_'+'_'.join(map(str,params)), rep_all_list)
	np.save(result_numpy_file, rep_all_list)

