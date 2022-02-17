main(){
python viz/expt_4_distributed_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd.pkl \
    --ours \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
    --loss-type ce ce ce ce \
    --models MNIST:IID/SignSGD MNIST:Non-IID/SignSGD FMNIST:IID/SignSGD FMNIST:Non-IID/SignSGD \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 1 1 0.9 0.8 \
    --ylim2 45 45 45 45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_4_distributed_main_with_acc_comm_curve
}

main_iid(){
python viz/expt_4_distributed_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    --ours \
	../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd_lbgm_0.6.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.6.pkl \
    --loss-type ce ce ce mse --wspace 0.45 \
    --models MNIST:IID/SignSGD FMNIST:IID/SignSGD CIFAR-10:IID/SignSGD CelebA:SignSGD \
    --m-int 1e8 1e8 1e8 1e8 \
    --m-str 10^8 10^8 10^8 \
    --u-int 0 0 0 \
    --u-str na na na \
    --ylim1 1 0.9 0.7 \
    --ylim2 0.45 0.45 3.5 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_4_distributed_main_iid_with_acc_comm_curve
}

main_non_iid(){
python viz/expt_4_distributed_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_signsgd.pkl \
    --ours \
	../ckpts_frankie/mnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts_frankie/fmnist_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts/cifar_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.2.pkl \
        ../ckpts/celeba_100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_100_lr_2e-05_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
    --loss-type ce ce ce mse --wspace 0.45 \
    --models MNIST:Non-IID/SignSGD FMNIST:Non-IID/SignSGD CIFAR-10:Non-IID/SignSGD CelebA:SignSGD \
    --m-int 1e6 1e6 1e6 1e6\
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 1e4\
    --u-str na na na 10^4\
    --ylim1 1 0.9 0.5 2\
    --ylim2 45 45 400 200 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_4_distributed_main_non_iid_with_acc_comm_curve
}


fcn(){
python viz/expt_4_distributed_with_acc_comm_curve.py --baseline \
    ../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd.pkl \
    --ours \
	../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts_frankie/mnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_10_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts_frankie/fmnist_100/history/clf_fcn_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.0003_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
    --loss-type ce ce ce ce \
    --models MNIST:IID/SignSGD MNIST:Non-IID/SignSGD FMNIST:IID/SignSGD FMNIST:Non-IID/SignSGD \
    --m-int 1e6 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 10^6 \
    --u-int 0 0 0 0 \
    --u-str na na na na \
    --ylim1 1 0.8 0.9 0.8 \
    --ylim2 1 1 1 1 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_4_distributed_fcn_with_acc_comm_curve
}

pfl(){
python viz/expt_4_distributed_with_acc_comm_curve.py --baseline \
    ../ckpts/cifar_100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_100_lr_0.001_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.01_decay_1e-05_batch_0_signsgd.pkl \
    ../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_0.0001_decay_1e-05_batch_0_signsgd.pkl \
	--ours \
        ../ckpts/cifar_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_3_num_workers_10_lr_0.001_decay_1e-05_batch_0_signsgd_lbgm_0.2.pkl \
        ../ckpts/cifar100_50/history/clf_resnet18_optim_sgd_uniform_True_non_iid_10_num_workers_50_lr_0.01_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
	../ckpts/celeba_10/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_10_lr_0.0001_decay_1e-05_batch_0_signsgd_lbgm_0.4.pkl \
    --loss-type ce ce mse \
    --models CIFAR-10:Non-IID/SignSGD CIFAR-100:Non-IID/SignSGD CelebA:SignSGD \
    --m-int 1e6 1e6 1e6 \
    --m-str 10^6 10^6 10^6 \
    --u-int 0 0 1e4\
    --u-str na na 10^4\
    --ylim1 0.8 0.65 2 \
    --ylim2 150 350 140 \
    --wspace 0.45 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/expt_4_distributed_resnet18_pfl_with_acc_comm_curve
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

