# vgg11 has batchsize 25000 instead of 50000 due to memory constraints

mnist_gd(){
    python process_grads.py --h \
	../ckpts/mnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_60000.pkl \
	../ckpts/mnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
	../ckpts/mnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
        ../ckpts/mnist/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_25000.pkl \
        ../ckpts/mnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
        --m 0 --dry-run 1
}


mnist_sgd(){
    python process_grads.py --h \
	../ckpts/mnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_128.pkl \
	../ckpts/mnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/mnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
        ../ckpts/mnist/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
        ../ckpts/mnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
        --m 0 --dry-run 1
}


cifar_gd(){
    # ../ckpts/cifar/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_60000.pkl \
    # ../ckpts/cifar/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \ done
    python process_grads.py --h \
	../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
	../ckpts/cifar/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_25000.pkl \
	../ckpts/cifar/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
        --m 0.9 --dry-run 1
}


cifar_sgd(){
    python process_grads.py --h \
	../ckpts/cifar/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/cifar/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
        --m 0.9 --dry-run 1
}


fmnist_gd(){
    python process_grads.py --h \
	../ckpts/fmnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_60000.pkl \
	../ckpts/fmnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
	../ckpts/fmnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
	../ckpts/fmnist/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_25000.pkl \
	../ckpts/fmnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_60000.pkl \
	--m 0.9 --dry-run 1
}


fmnist_sgd(){
    python process_grads.py --h \
	../ckpts/fmnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	../ckpts/fmnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
	--m 0.9 --dry-run 1
}

celeba_sgd(){
    python process_grads.py --h \
	../ckpts/celeba/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_vgg11_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	../ckpts/celeba/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
	--m 0.9 --dry-run 1
        
}



$1
