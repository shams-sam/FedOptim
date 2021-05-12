cifar(){
python viz/prelim_3.py --h \
    ../ckpts/cifar/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_0.01_decay_1e-05_batch_128.pkl \
    --dataset CIFAR-10 \
    --dry-run 0 --final 1 \
    --save ../ckpts/plots/prelim_3_cifar_cnn_128
}

celeba(){
python viz/prelim_3.py --h \
    ../ckpts/celeba/history/clf_cnn_optim_sgd_uniform_True_non_iid_0_num_workers_0_lr_1e-05_decay_1e-05_batch_256.pkl \
    --dataset CelebA \
    --dry-run 0 --final 1 \
    --save ../ckpts/plots/prelim_3_celeba_cnn_256
}


$1
