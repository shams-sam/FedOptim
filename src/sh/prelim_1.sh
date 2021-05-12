cifar(){
python viz/prelim_1.py --h \
    ../ckpts/cifar/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --models CIFAR-10:FCN CIFAR-10:CNN CIFAR-10:ResNet18 CIFAR-10:VGG19 \
    --loss-type ce ce ce ce\
    --dry-run 0 --final 1 \
    --ylim1 200 200 200 200 \
    --ylim2 1 1 1 1 \
    --save ../ckpts/plots/prelim_1_cifar_128
}


celeba(){
python viz/prelim_1.py --h \
    ../ckpts/celeba/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    ../ckpts/celeba/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_1e-05_decay_1e-05_batch_256.pkl \
    ../ckpts/celeba/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    ../ckpts/celeba/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    --models CelebA:FCN CelebA:CNN CelebA:ResNet18 CelebA:VGG19 \
    --loss-type mse mse mse mse\
    --dry-run 0 --final 1 \
    --ylim1 20 20 20 20 \
    --ylim2 500 50 50 50 \
    --save ../ckpts/plots/prelim_1_celeba_256
}


$1
