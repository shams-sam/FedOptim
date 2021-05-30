main(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    --loss-type ce ce ce ce\
    --models MNIST:IID MNIST:Non-IID FMNIST:IID FMNIST:Non-IID \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 1 1 0.8 0.8 \
    --ylim2 45 45 45 45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_main
}


non_iid(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_0/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
	../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_lbgm_0.9.pkl \
    --loss-type ce ce mse\
    --models MNIST:CNN FMNIST:CNN CelebA:CNN\
    --m-int 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6\
    --u-int 0 0 1e4\
    --u-str na na 10^4\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_non_iid
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
