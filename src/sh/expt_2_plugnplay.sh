main(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
    --loss-type ce ce ce ce ce ce ce ce \
    --models MNIST:IID/Top-K MNIST:Non-IID/Top-K FMNIST:IID/Top-K FMNIST:Non-IID/Top-K MNIST:IID/ATOMO MNIST:Non-IID/ATOMO FMNIST:IID/ATOMO FMNIST:Non-IID/ATOMO \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 0 0 0 0 \
    --u-str na na na na na na na na \
    --xlim 0 0 0 0 0 0 0 0 \
    --ylim1 1 1 0.9 0.9 0.9 0.9 0.8 0.8 \
    --ylim2 5 5 5 5 0.6 0.6 0.6 0.6 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_main
}

main_non_iid(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-06_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.2_residual.pkl \
	../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.2.pkl \
	../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
    --loss-type ce ce ce mse ce ce ce mse \
    --models MNIST:Non-IID/Top-K FMNIST:Non-IID/Top-K CIFAR-10:Non-IID/Top-K CelebA:Top-K MNIST:Non-IID/ATOMO FMNIST:Non-IID/ATOMO CIFAR-10:Non-IID/ATOMO CelebA:ATOMO \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 1e4 0 0 0 1e4 \
    --u-str na na na 10^4 na na na 10^4 \
    --xlim 0 0 500 300 0 0 500 300 \
    --ylim1 1 0.9 0.7 2 0.9 0.9 0.6 2 \
    --ylim2 5 5 40 20 0.6 0.6 4 2 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_main_non_iid
}

main_iid(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.2_residual.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.2.pkl \
    --loss-type ce ce ce ce ce ce \
    --models MNIST:IID/Top-K FMNIST:IID/Top-K CIFAR-10:IID/Top-K MNIST:IID/ATOMO FMNIST:IID/ATOMO CIFAR-10:IID/ATOMO \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 0 0 \
    --u-str na na na na na na \
    --xlim 0 0 500 0 0 500 \
    --ylim1 1 0.9 0.7 0.9 0.9 0.5 \
    --ylim2 5 5 40 0.6 0.6 4 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_main_iid
}


cnn(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.8.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.8.pkl \
    --loss-type ce ce ce ce \
    --models CIFAR-10:IID/Top-K CIFAR-10:Non-IID/Top-K CIFAR-10:IID/ATOMO CIFAR-10:Non-IID/ATOMO \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 0.7 0.7 0.6 0.6 \
    --ylim2 40 40 5 5 \
    --xlim 500 500 0 0 0 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_cnn
}


fcn(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
	../ckpts/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
	../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
	../ckpts/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.6.pkl \
    --loss-type ce ce ce ce ce ce ce ce \
    --models MNIST:IID/Top-K MNIST:Non-IID/Top-K FMNIST:IID/Top-K FMNIST:Non-IID/Top-K MNIST:IID/ATOMO MNIST:Non-IID/ATOMO FMNIST:IID/ATOMO FMNIST:Non-IID/ATOMO \
    --m-int 1e5 1e5 1e5 1e5 1e5 1e5 1e5 1e5 \
    --m-str 10^5 10^5 10^5 10^5 10^5 10^5 10^5 10^5 \
    --u-int 0 0 0 0 0 0 0 0 \
    --u-str na na na na na na na na\
    --ylim1 1 1 1 1 1 1 1 1 \
    --ylim2 1 1 1 1 2 2 2 2 \
    --xlim 0 0 0 0 0 0 0 0 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_fcn
}

resnet18(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_2e-05_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    --ours \
        ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/fmnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/mnist_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
        ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
	../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr2e-05_decay_1e-05_batch_0_topk_0.1_lbgm_0.8_residual.pkl \
    --loss-type ce ce ce ce ce ce ce ce \
    --models FMNIST:IID/Top-K FMNIST:Non-IID/Top-K MNIST:IID/Top-K MNIST:Non-IID/Top-K FMNIST:IID/ATOMO FMNIST:Non-IID/ATOMO MNIST:IID/ATOMO MNIST:Non-IID/ATOMO \
    --m-int 1e5 1e5 1e5 1e5 1e5 1e5 1e5 1e5 \
    --m-str 10^5 10^5 10^5 10^5 10^5 10^5 10^5 10^5 \
    --u-int 0 0 0 0 0 0 0 0 \
    --u-str na na na na na na na na\
    --ylim1 1 1 1 1 1 1 1 1 \
    --ylim2 1 1 1 1 2 2 2 2 \
    --xlim 0 0 0 0 0 0 0 0 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_resnet18
}

pfl(){
python viz/expt_2_plugnplay.py --baseline \
    ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.1_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_1e-05_decay_1e-05_batch_0_topk_0.1_residual.pkl \
    ../ckpts/cifar_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.1_decay_1e-05_batch_0_atomo_2.pkl \
    ../ckpts/celeba_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_atomo_2.pkl \
    --ours \
        ../ckpts/cifar_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_topk_0.1_lbgm_0.6_residual.pkl \
        ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.1_decay_1e-05_batch_0_topk_0.1_lbgm_0.6_residual.pkl \
        ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_1e-05_decay_1e-05_batch_0_topk_0.1_lbgm_0.4_residual.pkl \
	../ckpts/cifar_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_atomo_2_lbgm_0.2.pkl \
        ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.1_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
        ../ckpts/celeba_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_atomo_2_lbgm_0.4.pkl \
    --loss-type ce ce mse ce ce mse \
    --models CIFAR-10:Non-IID/Top-K CIFAR-100:Non-IID/Top-K CelebA:Top-K CIFAR-10:Non-IID/ATOMO CIFAR-100:Non-IID/ATOMO CelebA:ATOMO \
    --m-int 1e6 1e6 1e6 1e6 1e6 1e6\
    --m-str 10^6 10^6 10^6 10^6 10^6 10^6 \
    --u-int 0 0 1e4 0 0 1e4 \
    --u-str na na 10^4 na na 10^4 \
    --ylim1 0.8 0.6 2 0.8 0.6 2 \
    --ylim2 15 35 8 10 10 2 \
    --xlim 0 0 0 0 0 0 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_2_plugnplay_resnet18_pfl
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
