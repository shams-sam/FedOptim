################################################################################
# celeba - 0.5 subset:
################################################################################
celeba_cnn(){
python viz/prelim_2.py --h \
    ../ckpts/celeba/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_1e-05_decay_1e-05_batch_256.pkl \
    --dataset CelebA \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_celeba_cnn_256
}

celeba_fcn(){
python viz/prelim_2.py --h \
    ../ckpts/celeba/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    --dataset CelebA --cols 2 --rows 1\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_celeba_fcn_256
}

celeba_resnet18(){
python viz/prelim_2.py --h \
    ../ckpts/celeba/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    --dataset CelebA --cols 7 --rows 9 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_celeba_resnet18_256
}

celeba_vgg19(){
python viz/prelim_2.py --h \
    ../ckpts/celeba/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    --dataset CelebA --cols 7 --rows 10\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_celeba_vgg19_256
}


################################################################################
# cifar:
################################################################################
cifar_cnn(){
python viz/prelim_2.py --h \
    ../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset CIFAR-10 --cols 4 --rows 2\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_cifar_cnn_128
}

cifar_fcn(){
python viz/prelim_2.py --h \
    ../ckpts/cifar/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset CIFAR-10 --cols 2 --rows 1\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_cifar_fcn_128
}

cifar_resnet18(){
python viz/prelim_2.py --h \
    ../ckpts/cifar/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset CIFAR-10 --cols 7 --rows 9\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_cifar_resnet18_128
}

cifar_vgg19(){
python viz/prelim_2.py --h \
    ../ckpts/cifar/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset CIFAR-10 --cols 7 --rows 10\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_cifar_vgg19_128
}


################################################################################
# coco:
################################################################################
coco_unet(){
python viz/prelim_2.py --h \
    ../ckpts/coco/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_512.pkl \
    --dataset COCO --cols 7 --rows 10\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_coco_unet_512
}


################################################################################
# fmnist:
################################################################################
fmnist_cnn(){
python viz/prelim_2.py --h \
    ../ckpts/fmnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset FMNIST --cols 4 --rows 2\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_fmnist_cnn_128
}

fmnist_fcn(){
python viz/prelim_2.py --h \
    ../ckpts/fmnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset FMNIST --cols 2 --rows 1\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_fmnist_fcn_128
}

fmnist_resnet18(){
python viz/prelim_2.py --h \
    ../ckpts/fmnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    --dataset FMNIST --cols 7 --rows 9\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_fmnist_resnet18_256
}

fmnist_vgg19(){
python viz/prelim_2.py --h \
    ../ckpts/fmnist/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    --dataset FMNIST --cols 7 --rows 10\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_fmnist_vgg19_256
}


################################################################################
# mnist:
################################################################################
mnist_cnn(){
python viz/prelim_2.py --h \
    ../ckpts/mnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset MNIST --cols 4 --rows 2\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_mnist_cnn_128
}

mnist_fcn(){
python viz/prelim_2.py --h \
    ../ckpts/mnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset MNIST --cols 2 --rows 1\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_mnist_fcn_128
}

mnist_resnet18(){
python viz/prelim_2.py --h \
    ../ckpts/mnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    --dataset MNIST --cols 7 --rows 9\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_mnist_resnet18_256
}

mnist_vgg19(){
python viz/prelim_2.py --h \
    ../ckpts/mnist/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    --dataset MNIST --cols 7 --rows 10\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_mnist_vgg19_256
}

################################################################################
# voc:
################################################################################
voc_unet(){
python viz/prelim_2.py --h \
    ../ckpts/voc/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_32.pkl \
    --dataset PascalVOC --cols 7 --rows 10\
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_voc_unet_32
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
