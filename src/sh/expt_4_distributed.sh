main(){
python viz/expt_4_distributed.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    --ours \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.6.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.8.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.6.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.8.pkl \
    --loss-type ce ce ce ce \
    --models MNIST:IID/SignSGD MNIST:Non-IID/SignSGD FMNIST:IID/SignSGD FMNIST:Non-IID/SignSGD \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 1 1 0.9 0.8 \
    --ylim2 45 45 45 45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_4_distributed_main
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

