# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
#        clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_1_residual.pkl \
#        clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_residual.pkl \
#        --labels 'sgd' 'kgrad=1' 'topk=0.1' 'topk=0.1, kgrad=1' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.jpg


# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
#        clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_1_residual.pkl \
#        clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_residual.pkl \
#        --labels 'sgd' 'kgrad=1' 'topk=0.1' 'topk=0.1, kgrad=1' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.jpg

# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
#        clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1_residual.pkl \
#        clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_residual.pkl \
#        --labels 'sgd' 'kgrad=1' 'topk=0.1' 'topk=0.1, kgrad=1' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.jpg


# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
#        clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1_residual.pkl \
#        clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
#        clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_residual.pkl \
#        --labels 'sgd' 'kgrad=1' 'topk=0.1' 'topk=0.1, kgrad=1' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_20.pkl \
       --labels 'sgd' 'kgrad=1' 'kgrad=5' 'kgrad=10' 'kgrad=20' --ncols 3 \
       --name comparison_clf_fcn_paradigm_kgrad_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_20.pkl \
       --labels 'sgd' 'kgrad=1' 'kgrad=5' 'kgrad=10' 'kgrad=20' --ncols 3 \
       --name comparison_clf_fcn_paradigm_kgrad_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_20.pkl \
       --labels 'sgd' 'topk=0.1' 'kgrad=1,topk=0.1' 'kgrad=5,topk=0.1' 'kgrad=10,topk=0.1' 'kgrad=20,topk=0.1' --ncols 3 \
       --name comparison_clf_fcn_paradigm_topk_kgrad_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_20.pkl \
       --labels 'sgd' 'topk=0.1' 'kgrad=1,topk=0.1' 'kgrad=5,topk=0.1' 'kgrad=10,topk=0.1' 'kgrad=20,topk=0.1' --ncols 3 \
       --name comparison_clf_fcn_paradigm_topk_kgrad_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_20.pkl \
       --labels 'sgd' 'kgrad=1' 'kgrad=5' 'kgrad=10' 'kgrad=20' --ncols 3 \
       --name comparison_clf_fcn_paradigm_kgrad_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_20.pkl \
       --labels 'sgd' 'kgrad=1' 'kgrad=5' 'kgrad=10' 'kgrad=20' --ncols 3 \
       --name comparison_clf_fcn_paradigm_kgrad_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_20.pkl \
       --labels 'sgd' 'topk=0.1' 'kgrad=1,topk=0.1' 'kgrad=5,topk=0.1' 'kgrad=10,topk=0.1' 'kgrad=20,topk=0.1' --ncols 3 \
       --name comparison_clf_fcn_paradigm_topk_kgrad_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_5_10_20.jpg

python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
       clf_fcn_noise_None_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_None_paradigm_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_5.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_10.pkl \
       clf_fcn_noise_None_paradigm_kgrad_sgd_topk_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_20.pkl \
       --labels 'sgd' 'topk=0.1' 'kgrad=1,topk=0.1' 'kgrad=5,topk=0.1' 'kgrad=10,topk=0.1' 'kgrad=20,topk=0.1' --ncols 3 \
       --name comparison_clf_fcn_paradigm_topk_kgrad_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_kgrad_1_5_10_20.jpg
