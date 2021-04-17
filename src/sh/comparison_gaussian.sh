mnist(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 51 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_1e-05.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_0.0001.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_0.001.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_0.01.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_0.1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_1.0.pkl \
	   --labels 'fedavg' '$\sigma^2$=0.00001' '$\sigma^2$=0.0001' '$\sigma^2$=0.001' '$\sigma^2$=0.01' '$\sigma^2$=0.1' '$\sigma^2$=1.0' \
	   --ncols 4 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_varying.jpg

    # python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 51 --histories \
    # 	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    # 	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_0.01.pkl \
    # 	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_0.1.pkl \
    # 	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_1.0.pkl \
    # 	   --labels 'fedavg' 'var=0.01' 'var=0.1' 'var=1.0' --ncols 2 \
    # 	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_noise_varying.jpg

}



$1
