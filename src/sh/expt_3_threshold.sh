main(){
python viz/expt_3_threshold.py --history \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce ce \
    --break-pt 0 6 12 18 24 \
    --models MNIST:IID MNIST:Non-IID FMNIST:IID FMNIST:Non-IID \
    --labels na 0.9 0.8 0.6 0.4 0.2 \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 1 1 1 1 \
    --ylim2 45 45 45 45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_3_threshold_main
}


cnn(){
python viz/expt_3_threshold.py --history \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0.pkl \ # training another set using 1e-05
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce mse \
    --break-pt 0 6 12 18 \
    --models CIFAR-10:IID CIFAR-10:Non-IID CelebA:Regression\
    --labels na 0.9 0.8 0.6 0.4 0.2 \
    --m-int 1e8 1e8 1e8 \
    --m-str 10^8 10^8 10^8 \
    --u-int 0 0 1e4 \
    --u-str na na 10^4\
    --ylim1 0.75 0.75 5 \
    --ylim2 4 4 2 \
    --wspace 0.45 --ncol 2 \
    --leg-w 4.0 --leg-h 1.3 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_3_threshold_cnn
}



fcn(){
python viz/expt_3_threshold.py --history \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce ce \
    --break-pt 0 6 12 18 24 \
    --models MNIST:IID MNIST:Non-IID FMNIST:IID FMNIST:Non-IID\
    --labels na 0.9 0.8 0.6 0.4 0.2 \
    --m-int 1e5 1e5 1e5 1e5 \
    --m-str 10^5 10^5 10^5 10^5 \
    --u-int 0 0 0 0 \
    --u-str na na na na\
    --ylim1 1 1 1 1 \
    --ylim2 10 10 10 10 \
    --wspace 0.45 --ncol 2 \
    --leg-w 5.4 --leg-h 1.3 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_3_threshold_fcn
}


resnet18(){
python viz/expt_3_threshold.py --history \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.8.pkl \
    na \
    na \
    na \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0_lbgm_0.9.pkl \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0_lbgm_0.8.pkl \
    na \
    na \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce ce ce ce mse\
    --break-pt 0 6 12 18 24 30 36 42\
    --models FMNIST:IID FMNIST:Non-IID MNIST:IID MNIST:Non-IID CIFAR-10:IID CIFAR-10:Non-IID CelebA:Regression\
    --labels na 0.9 0.8 0.6 0.4 0.2 \
    --m-int 1e9 1e9 1e9 1e9 1e9 1e9 1e9 \
    --m-str 10^9 10^9 10^9 10^9 10^9 10^9 10^9\
    --u-int 0 0 0 0 0 0 1e4\
    --u-str na na na na na na 10^4\
    --ylim1 1 0.6 1 1 0.6 0.6 2.0\
    --ylim2 2 2 2 2 3 3 3 \
    --wspace 0.5 --ncol 3 \
    --leg-w 10.0 --leg-h 1.0 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_3_threshold_resnet18
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
