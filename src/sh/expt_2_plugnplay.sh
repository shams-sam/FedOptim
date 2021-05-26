main(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/celeba_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_0.0001_decay_1e-05_batch_0_topk_0.01.pkl \
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
