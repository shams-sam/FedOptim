# python comparison.py --dataset mnist --num-nodes 1 --dpi 100 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_1_lr_0.006_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_1_lr_0.006_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_1_lr_0.006_decay_1e-05_batch_0.pkl \
#        --labels  'sgd' 'adam' 'conj' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_1_lr_0.006_decay_1e-05_batch_0.jpg

# python comparison.py --dataset mnist --num-nodes 10 --dpi 100 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.pkl \
#        --labels  'sgd' 'adam' 'conj' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.jpg

# python comparison.py --dpi 100 --infer-folder 0 --histories \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_1/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_1_lr_0.006_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_10_lr_0.006_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_100/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_100_lr_0.006_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_500/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_500_lr_0.006_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj@1' 'conj@10' 'conj@100' 'conj@500' --ncols 3\
#        --name ../ckpts/plots/comparison_clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_varying_lr_0.006_decay_1e-05_batch_0.jpg

# python comparison.py --dpi 100 --infer-folder 0 --histories \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj@10' --ncols 3\
#        --name ../ckpts/plots/comparison_clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_varying_lr_0.01_decay_1e-05_batch_0.jpg

# python comparison.py --dpi 100 --infer-folder 0 --histories \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_1/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_10/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_100/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        ../ckpts/mnist_500/history/clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_500_lr_0.1_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj@10' --ncols 3\
#        --name ../ckpts/plots/comparison_clf_fcn_paradigm_conj_uniform_True_non_iid_10_num_workers_varying_lr_0.1_decay_1e-05_batch_0.jpg


# # lr 0.01
# # iid on 1
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 1 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.jpg

# # iid on 10
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.jpg

# # iid on 100
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.jpg

# # non-iid on 10
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.jpg

# # non-iid on 100
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.jpg


# ## lr 0.1
# # iid on 1
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 1 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_1_lr_0.1_decay_1e-05_batch_0.jpg

# # iid on 10
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_10_lr_0.1_decay_1e-05_batch_0.jpg

# # iid on 100
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_10_num_workers_100_lr_0.1_decay_1e-05_batch_0.jpg

# # non-iid on 10
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_1_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_1_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_1_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_1_num_workers_10_lr_0.1_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_1_num_workers_10_lr_0.1_decay_1e-05_batch_0.jpg

# # non-iid on 100
# python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
#        clf_fcn_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_adam_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_pr_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_conj_fr_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_orth_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_0.pkl \
#        --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3 \
#        --name comparison_clf_fcn_paradigm_varying_uniform_True_non_iid_1_num_workers_100_lr_0.1_decay_1e-05_batch_0.jpg



## noisy
# iid on 1
python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 1 --histories \
       clf_fcn_noise_True_paradigm_sgd_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_adam_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_orth_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.pkl \
       --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
       --name comparison_clf_fcn_noise_True_paradigm_varying_uniform_True_non_iid_10_num_workers_1_lr_0.01_decay_1e-05_batch_0.jpg

# iid on 10
python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
       clf_fcn_noise_True_paradigm_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_adam_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_orth_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
       --name comparison_clf_fcn_noise_True_paradigm_varying_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.jpg

# iid on 100
python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
       clf_fcn_noise_True_paradigm_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_adam_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_pr_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_fr_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_orth_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3 \
       --name comparison_clf_fcn_noise_True_paradigm_varying_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.jpg

# non-iid on 10
python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 10 --histories \
       clf_fcn_noise_True_paradigm_sgd_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_adam_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_pr_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_fr_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_orth_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
       --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3\
       --name comparison_clf_fcn_noise_True_paradigm_varying_uniform_True_non_iid_1_num_workers_10_lr_0.01_decay_1e-05_batch_0.jpg

# non-iid on 100
python comparison.py --dpi 100 --infer-folder 1 --dataset mnist --num-nodes 100 --histories \
       clf_fcn_noise_True_paradigm_sgd_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_adam_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_pr_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_conj_fr_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_noise_True_paradigm_orth_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
       --labels 'sgd' 'adam' 'conj w/ pr' 'conj w/ fr' 'orth' --ncols 3 \
       --name comparison_clf_fcn_noise_True_paradigm_varying_uniform_True_non_iid_1_num_workers_100_lr_0.01_decay_1e-05_batch_0.jpg
