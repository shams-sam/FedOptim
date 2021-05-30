main(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.6_residual.pkl \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.6_residual.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.8.pkl \
    --loss-type ce ce ce ce ce ce ce ce \
    --models MNIST:IID/Top-K MNIST:Non-IID/Top-K FMNIST:IID/Top-K FMNIST:Non-IID/Top-K MNIST:IID/ATOMO MNIST:Non-IID/ATOMO FMNIST:IID/ATOMO FMNIST:Non-IID/ATOMO \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 0 0 0 0 \
    --u-str na na na na na na na na \
    --ylim1 1 1 0.8 0.8 0.9 0.9 0.8 0.8 \
    --ylim2 5 5 5 5 0.6 0.6 0.6 0.6 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_main
}


bkp(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.6_residual.pkl \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.6_residual.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.6.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.8.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.6.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.8.pkl \
    --loss-type ce ce ce ce ce ce ce ce \
    --models MNIST:IID/Top-K MNIST:Non-IID/Top-K FMNIST:IID/Top-K FMNIST:Non-IID/Top-K MNIST:IID/SignSGD MNIST:Non-IID/SignSGD FMNIST:IID/SignSGD FMNIST:Non-IID/SignSGD \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 0 0 0 0 \
    --u-str na na na na na na na na \
    --ylim1 1 1 0.8 0.8 1 1 0.8 0.8 \
    --ylim2 5 5 5 5 45 45 45 45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_main
}



non_iid(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/fmnist_100/history/
    ../ckpts/mnist_100/history/
    ../ckpts/celeba_100/history/
    --ours \
	../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.8.pkl \
	../ckpts/celeba_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_0.0001_decay_1e-05_batch_0_topk_0.01_lbgm_0.8.pkl \
    --loss-type ce ce mse \
    --models FMNIST:TopK MNIST:SignSGD CelebA:FCN\
    --m-int 1e4 1e4 1e4\
    --m-str 10^4 10^4 10^4 \
    --u-int 0 0 1e3\
    --u-str na na 10^3\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_main
}




if [ $2 = 'f' ]; then
    final=1
    dry=0
elif [ $2 = 'd' ]; then
    final=0
    dry=1
else
    final=0
    dry=0
fi


$1
