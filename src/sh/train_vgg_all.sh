proc1(){
    sh sh/train_federated_vgg19.sh celeba 0
    sh sh/train_federated_vgg19.sh cifar 10
    sh sh/train_federated_vgg19.sh cifar 3
    sh sh/train_lbgm_vgg19.sh celeba 0
    sh sh/train_lbgm_vgg19.sh cifar 10
    sh sh/train_lbgm_vgg19.sh cifar 3
    sh sh/train_signsgd_vgg19.sh celeba 0
    sh sh/train_signsgd_vgg19.sh cifar 10
    sh sh/train_signsgd_vgg19.sh cifar 3
}

proc2(){
    sh sh/train_topk_vgg19.sh celeba 0 0.1
    sh sh/train_topk_vgg19.sh cifar 10 0.1
    sh sh/train_topk_vgg19.sh cifar 3 0.1
    sh sh/train_topk_lbgm_vgg19.sh celeba 0 0.1
    sh sh/train_topk_lbgm_vgg19.sh cifar 10 0.1
    sh sh/train_topk_lbgm_vgg19.sh cifar 3 0.1
    sh sh/train_signsgd_lbgm_vgg19.sh celeba 0
    sh sh/train_signsgd_lbgm_vgg19.sh cifar 10
    sh sh/train_signsgd_lbgm_vgg19.sh cifar 3
}

$1
