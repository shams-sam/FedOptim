main(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce ce\
    --models MNIST:IID MNIST:Non-IID FMNIST:IID FMNIST:Non-IID \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 1 1 1 1 \
    --ylim2 45 45 45 45 \
    --xlim 0 0 0 0 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_main
}

main_non_iid(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.1.pkl \
	../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce mse \
    --models MNIST:Non-IID FMNIST:Non-IID CIFAR-10:Non-IID CelebA:Regression \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 1e4 \
    --u-str na na na 10^4 \
    --ylim1 1 0.9 0.7 2 \
    --ylim2 45 45 400 150 \
    --xlim 0 0 500 200 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_main_non_iid
}

# cifar lbgm 0.1 is available to reduce performance gap
main_iid(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce mse \
    --models MNIST:IID FMNIST:IID CIFAR-10:IID \
    --m-int 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 \
    --u-int 0 0 0 \
    --u-str na na na \
    --ylim1 1 0.9 0.7 \
    --ylim2 45 45 400 \
    --xlim 0 0 500 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_main_iid
}


cnn(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
        ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_lbgm_0.4.pkl \
    --loss-type ce ce mse\
    --models CIFAR-10:IID CIFAR-10:Non-IID CelebA:Regression \
    --m-int 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 1e4 \
    --u-str na na 10^4 \
    --ylim1 0.7 0.7 4 \
    --ylim2 400 400 120 \
    --xlim 500 500 150 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_cnn
}

fcn(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce ce \
    --models MNIST:IID MNIST:Non-IID FMNIST:IID FMNIST:Non-IID \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0  \
    --u-str na na na na \
    --ylim1 1 1 0.9 0.9 \
    --ylim2 1 1 1 1 \
    --xlim 0 0 0 0 \
    --wspace 0.35 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_fcn
}

resnet18(){
python viz/expt_1_standalone.py --baseline \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
        ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
        ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
	../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0_lbgm_0.8.pkl \
    --loss-type ce ce ce ce ce ce mse \
    --models FMNIST:IID FMNIST:Non-IID MNIST:IID MNIST:Non-IID CIFAR-10:IID CIFAR-10:Non-IID CelebA:Regression \
    --m-int 1e9 1e9 1e9 1e9 1e9 1e9 1e9 \
    --m-str 10^9 10^9 10^9 10^9 10^9 10^9 10^9 \
    --u-int 0 0 0 0 0 0 1e4  \
    --u-str na na na na na na 10^4\
    --ylim1 1 0.6 1 1 0.6 1 1.5\
    --ylim2 2 2 2 2 3 3 3 \
    --xlim 0 0 0 0 0 0 200 \
    --wspace 0.5 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_resnet18
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
