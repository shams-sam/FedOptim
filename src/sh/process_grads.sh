################################################################################
# celeba - 0.5 subset: done
################################################################################
celeba(){
    python process_grads.py --h \
	../ckpts/celeba/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_1e-05_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	--m 0.9 --dry-run $dry
}


################################################################################
# cifar: done
################################################################################
cifar(){
    python process_grads.py --h \
	../ckpts/cifar/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	--m 0.9 --dry-run $dry
}


################################################################################
# coco: todo
################################################################################
coco(){
    python process_grads.py --h \
	../ckpts/coco/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_512.pkl \
        --m 0.9 --dry-run $dry
}


################################################################################
# fmnist: done
################################################################################
fmnist(){
    python process_grads.py --h \
	../ckpts/fmnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
	../ckpts/fmnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
	--m 0.9 --dry-run $dry
}


################################################################################
# mnist: done
################################################################################
mnist(){
    python process_grads.py --h \
	../ckpts/mnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
	../ckpts/mnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/mnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
        ../ckpts/mnist/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
        ../ckpts/mnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
        --m 0.9 --dry-run $dry
}


################################################################################
# voc: wip
################################################################################
voc(){
    python process_grads.py --h \
	../ckpts/voc/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_32.pkl \
        --m 0.9 --dry-run $dry
}



if [ $2 = 'd' ]; then
    dry=1
elif [ $2 = 'f' ]; then
    dry=0
fi


$1
