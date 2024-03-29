main(){
python viz/expt_1_standalone_with_acc_comm_tradeoff.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
        ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
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
    --save ../ckpts/plots/expt_1_standalone_main_with_acc_comm_curve
}

main_non_iid(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.1.pkl \
	../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_1e-05_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce mse \
    --models MNIST:Non-IID FMNIST:Non-IID CIFAR-10:Non-IID CelebA:Regression \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 1e4 \
    --u-str na na na 10^4 \
    --ylim1 1 0.9 0.7 2 \
    --ylim2 45 150 400 150 \
    --xlim 0 0 500 200 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_main_non_iid_with_acc_comm_curve
}

# cifar lbgm 0.1 is available to reduce performance gap
main_iid(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/voc_100/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_0.001_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
        ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.4.pkl \
        ../ckpts/voc_100/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_0.001_decay_1e-05_batch_0_lbgm_0.9.pkl \
    --loss-type ce ce ce mse \
    --models MNIST:IID FMNIST:IID CIFAR-10:IID PascalVOC:Segmentation\
    --m-int 1e6 1e6 1e6 1e6\
    --m-str 10^6 10^6 10^6 10^6\
    --u-int 0 0 0 1\
    --u-str na na na na \
    --ylim1 1 0.9 0.7 4 \
    --ylim2 45 45 400 600\
    --xlim 0 0 500 300\
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_main_iid_with_acc_comm_curve
}


cnn(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
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
    --save ../ckpts/plots/expt_1_standalone_cnn_with_acc_comm_curve
}

fcn(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.6.pkl \
        ../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
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
    --save ../ckpts/plots/expt_1_standalone_fcn_with_acc_comm_curve
}

resnet18(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
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
    --save ../ckpts/plots/expt_1_standalone_resnet18_with_acc_comm_curve
}


pfl(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
   ../ckpts/mnist_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
   ../ckpts/fmnist_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0.pkl \
   ../ckpts/cifar_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0.pkl \
   ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.1_decay_1e-05_batch_0.pkl \
   ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_1e-05_decay_1e-05_batch_0.pkl \
   --ours \
        ../ckpts/mnist_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/fmnist_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.1_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_1e-05_decay_1e-05_batch_0_lbgm_0.2.pkl \
    --loss-type ce ce ce ce mse \
    --models MNIST:Non-IID FMNIST:Non-IID CIFAR-10:Non-IID CIFAR-100:Non-IID CelebA:Regression \
    --m-int 1e8 1e8 1e8 1e8 1e8\
    --m-str 10^8 10^8 10^8 10^8 10^8\
    --u-int 0 0 0 0 1e4 \
    --u-str na na na na 10^4 \
    --ylim1 1 1 0.8 0.6 2 \
    --ylim2 1 1 2 4 1 \
    --xlim 0 0 0 0 0 \
    --wspace 0.5 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_resnet18_pfl_with_acc_comm_curve
}

sampled_non_iid(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_sampled_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_sampled_lr_1e-05_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_sampled_lr_0.03_decay_1e-05_batch_0_lbgm_0.2.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0_lbgm_0.05.pkl \
	../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_sampled_lr_1e-05_decay_1e-05_batch_0_lbgm_0.4.pkl \
    --loss-type ce ce ce mse\
    --models MNIST:Non-IID FMNIST:Non-IID CIFAR-10:Non-IID CelebA:Regression\
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 1e4 \
    --u-str na na na 10^4 \
    --ylim1 1 0.9 0.6 2 \
    --ylim2 45 45 400 200 \
    --xlim 0 0 0 0 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_sampled_non_iid_with_acc_comm_curve
}


sampled_iid(){
python viz/expt_1_standalone_with_acc_comm_curve.py --baseline \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_sampled_lr_0.03_decay_1e-05_batch_0.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0.pkl \
    --ours \
        ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0_lbgm_0.2.pkl \
        ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_sampled_lr_0.03_decay_1e-05_batch_0_lbgm_0.4.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_sampled_lr_0.01_decay_1e-05_batch_0_lbgm_0.1.pkl \
    --loss-type ce ce ce\
    --models MNIST:IID FMNIST:IID CIFAR-10:IID \
    --m-int 1e6 1e6 1e6\
    --m-str 10^6 10^6 10^6 \
    --u-int 0 0 0 \
    --u-str na na na \
    --ylim1 1 0.9 0.6 \
    --ylim2 45 45 400 200 \
    --xlim 0 0 0 0 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_1_standalone_sampled_iid_with_acc_comm_curve
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
