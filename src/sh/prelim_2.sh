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
    --dataset CelebA --cols 4 --rows 16 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_celeba_resnet18_256
}

celeba_vgg19(){
python viz/prelim_2.py --h \
    ../ckpts/celeba/history/clf_vgg19_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.001_decay_1e-05_batch_256.pkl \
    --dataset CelebA \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_celeba_vgg19_256
}


################################################################################
# cifar:
################################################################################
cifar(){
python viz/prelim_2.py --h \
    ../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset CIFAR-10 \
    --dry-run $dry --final $final \
    --save ../ckpts/plots/prelim_2_cifar_cnn_128
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
