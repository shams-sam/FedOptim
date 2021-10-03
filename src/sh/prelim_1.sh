################################################################################
# celeba - 0.5 subset: done
################################################################################
celeba(){
python viz/prelim_1.py --h \
    ../ckpts/celeba/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    ../ckpts/celeba/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_1e-05_decay_1e-05_batch_256.pkl \
    ../ckpts/celeba/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    ../ckpts/celeba/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    --models CelebA:FCN CelebA:CNN CelebA:ResNet18 CelebA:VGG19 \
    --loss-type mse mse mse mse\
    --dry-run $dry --final $final \
    --ylim1 20 20 20 20 \
    --ylim2 500 50 50 50 \
    --save ../ckpts/plots/prelim_1_celeba_256
}


################################################################################
# cifar: done
################################################################################
cifar(){
python viz/prelim_1.py --h \
    ../ckpts/cifar/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --models CIFAR-10:FCN CIFAR-10:CNN CIFAR-10:ResNet18 CIFAR-10:VGG19 \
    --loss-type ce ce ce ce\
    --dry-run $dry --final $final \
    --ylim1 200 200 200 200 \
    --ylim2 1 1 1 1 \
    --save ../ckpts/plots/prelim_1_cifar_128
}


################################################################################
# cifar100: done
################################################################################
cifar100(){
python viz/prelim_1.py --h \
    ../ckpts/cifar100/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.1_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar100/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.1_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar100/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.1_decay_1e-05_batch_128.pkl \
    ../ckpts/cifar100/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.1_decay_1e-05_batch_128.pkl \
    --models CIFAR100:FCN CIFAR100:CNN CIFAR100:ResNet18 CIFAR100:VGG19 \
    --loss-type ce ce ce ce\
    --dry-run $dry --final $final \
    --ylim1 200 200 200 200 \
    --ylim2 1 1 1 1 \
    --save ../ckpts/plots/prelim_1_cifar100_128
}


################################################################################
# fmnist: done
################################################################################
fmnist(){
python viz/prelim_1.py --h \
    ../ckpts/fmnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/fmnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/fmnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    ../ckpts/fmnist/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    --models FMNIST:FCN FMNIST:CNN FMNIST:ResNet18 FMNIST:VGG19 \
    --loss-type ce ce ce ce \
    --dry-run $dry --final $final \
    --ylim1 100 100 100 100 \
    --ylim2 1 1 1 1 \
    --save ../ckpts/plots/prelim_1_fmnist_256
}


################################################################################
# mnist: done
################################################################################
mnist(){
python viz/prelim_1.py --h \
    ../ckpts/mnist/history/clf_fcn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/mnist/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    ../ckpts/mnist/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    ../ckpts/mnist/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_256.pkl \
    --models MNIST:FCN MNIST:CNN MNIST:ResNet18 MNIST:VGG19 \
    --loss-type ce ce ce ce \
    --dry-run $dry --final $final \
    --ylim1 100 100 100 100 \
    --ylim2 1 1 1 1 \
    --save ../ckpts/plots/prelim_1_mnist_256
}

################################################################################
# imagenet: done
################################################################################
imagenet(){
python viz/prelim_1.py --h \
    ../ckpts/imagenet/history/clf_resnet18_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.1_decay_1e-05_batch_512.pkl \
    --models ImageNet:ResNet18 \
    --loss-type ce \
    --dry-run $dry --final $final \
    --ylim1 100 \
    --ylim2 1 --use-train 1 \
    --save ../ckpts/plots/prelim_1_imagenet_256
}


################################################################################
# svm: done
################################################################################
svm(){
python viz/prelim_1.py --h \
    ../ckpts/cifar/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
    ../ckpts/fmnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
    ../ckpts/mnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
    --models CIFAR:SVM FMNIST:SVM MNIST:SVM \
    --loss-type ce ce ce \
    --dry-run $dry --final $final \
    --ylim1 20 20 20 \
    --ylim2 1 1 1 \
    --save ../ckpts/plots/prelim_1_svm_128
}


################################################################################
# svm: done
################################################################################
svm(){
python viz/prelim_1.py --h \
    ../ckpts/cifar/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
    ../ckpts/fmnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
    ../ckpts/mnist/history/clf_svm_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_128.pkl \
    --models CIFAR:SVM FMNIST:SVM MNIST:SVM \
    --loss-type ce ce ce \
    --dry-run $dry --final $final \
    --ylim1 20 20 20 \
    --ylim2 1 1 1 \
    --save ../ckpts/plots/prelim_1_svm_128
}


################################################################################
# segmentation
################################################################################
seg(){
python viz/prelim_1.py --h \
    ../ckpts/coco/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.0001_decay_1e-05_batch_512.pkl \
    ../ckpts/voc/history/clf_unet_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_32.pkl \
    --models COCO:Segmentation PascalVOC:Segmentation \
    --loss-type celoss celoss \
    --dry-run $dry --final $final \
    --ylim1 50 50 \
    --ylim2 1 3 \
    --save ../ckpts/plots/prelim_1_seg_128
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
