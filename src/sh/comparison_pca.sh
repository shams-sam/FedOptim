mnist_var(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.95.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.99.pkl \
	   --labels 'fedavg' 'exp-var=0.95' 'exp-var=0.99' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.95.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.99.pkl \
	   --labels 'fedavg' 'exp-var=0.95' 'exp-var=0.99' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_varying.jpg
}

mnist(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_2.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_3.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_5.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_10.pkl \
	   --labels 'fedavg'  'nc=1' 'nc=2' 'nc=3' 'nc=5' 'nc=10' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_nc_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_2.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_3.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_5.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_10.pkl \
	   --labels 'fedavg' 'nc=1' 'nc=2' 'nc=3' 'nc=5' 'nc=10' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_nc_varying.jpg
}

mnist_sdir_full(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_1_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_2_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_3_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_5_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_10_full.pkl \
	   --labels 'fedavg'  'nc=1' 'nc=2' 'nc=3' 'nc=5' 'nc=10' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_nc_full_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_1_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_2_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_3_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_5_full.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_10_full.pkl \
	   --labels 'fedavg' 'nc=1' 'nc=2' 'nc=3' 'nc=5' 'nc=10' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_nc_full_varying.jpg
}


fmnist(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset fmnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.9.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.95.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.99.pkl \
	   --labels 'fedavg' 'exp-var=0.9' 'exp-var=0.95' 'exp-var=0.99' --ncols 4 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset fmnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.9.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.95.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_0.99.pkl \
	   --labels 'fedavg' 'exp-var=0.9' 'exp-var=0.95' 'exp-var=0.99' --ncols 4 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_varying.jpg
}

fmnist_nc(){
    python comparison.py --dpi 100 --infer-folder 1 --dataset fmnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_2.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_3.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_5.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_10.pkl \
	   --labels 'fedavg'  'nc=1' 'nc=2' 'nc=3' 'nc=5' 'nc=10' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_nc_varying.jpg

    python comparison.py --dpi 100 --infer-folder 1 --dataset fmnist --num-nodes 100 --epochs 50 --histories \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_1.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_2.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_3.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_5.pkl \
	   clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_10.pkl \
	   --labels 'fedavg' 'nc=1' 'nc=2' 'nc=3' 'nc=5' 'nc=10' --ncols 3 \
	   --name comparison_clf_fcn_optim_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_pca_nc_varying.jpg
}


$1
