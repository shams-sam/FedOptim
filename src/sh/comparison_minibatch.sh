mnist(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_4.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_64.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_256.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_512.pkl \
	   --labels 'B=1' 'B=4' 'B=16' 'B=64' 'B=256' 'B=512' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_4.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_64.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_256.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_512.pkl \
	   --labels 'B=1' 'B=4' 'B=16' 'B=64' 'B=256' 'B=512' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_varying.jpg
}

cifar(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset cifar --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_4.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_16.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_64.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_256.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_512.pkl \
	   --labels 'B=1' 'B=4' 'B=16' 'B=64' 'B=256' 'B=512' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset cifar --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_4.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_16.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_64.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_256.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_512.pkl \
	   --labels 'B=1' 'B=4' 'B=16' 'B=64' 'B=256' 'B=512' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_varying.jpg
}

$1
