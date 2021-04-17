mnist(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1_tol_0.1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1_tol_0.2.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_100.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_1000.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_10000.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_100000.pkl \
	   --labels 'fedavg' 'rp=100' 'rp=1000' 'rp=10000' 'rp=100000' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_100.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_1000.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_10000.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_100000.pkl \
	   --labels 'fedavg' 'rp=100' 'rp=1000' 'rp=10000' 'rp=100000' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_rp_varying.jpg

}


$1
