central(){
    python make_table.py --dataset celeba --paradigm central
    python make_table.py --dataset cifar --paradigm central
    python make_table.py --dataset cifar100 --paradigm central
    python make_table.py --dataset fmnist --paradigm central
    python make_table.py --dataset mnist --paradigm central
    # python make_table.py --dataset imagenet --paradigm central
}

federated(){
    python make_table.py --dataset celeba --paradigm federated
    python make_table.py --dataset cifar --paradigm federated
    python make_table.py --dataset cifar100 --paradigm federated
    python make_table.py --dataset fmnist --paradigm federated
    python make_table.py --dataset mnist --paradigm federated
    # python make_table.py --dataset imagenet --paradigm federated
}

lbgm(){
    python make_table.py --dataset celeba --paradigm lbgm
    python make_table.py --dataset cifar --paradigm lbgm
    python make_table.py --dataset cifar100 --paradigm lbgm
    python make_table.py --dataset fmnist --paradigm lbgm
    python make_table.py --dataset mnist --paradigm lbgm
    # python make_table.py --dataset imagenet --paradigm lbgm
}


atomo(){
    python make_table.py --dataset celeba --paradigm atomo
    python make_table.py --dataset cifar --paradigm atomo
    python make_table.py --dataset cifar100 --paradigm atomo
    python make_table.py --dataset fmnist --paradigm atomo
    python make_table.py --dataset mnist --paradigm atomo
    # python make_table.py --dataset imagenet --paradigm atomo
}

atomo_lbgm(){
    python make_table.py --dataset celeba --paradigm atomo_lbgm
    python make_table.py --dataset cifar --paradigm atomo_lbgm
    python make_table.py --dataset cifar100 --paradigm atomo_lbgm
    python make_table.py --dataset fmnist --paradigm atomo_lbgm
    python make_table.py --dataset mnist --paradigm atomo_lbgm
    # python make_table.py --dataset imagenet --paradigm atomo_lbgm
}

signsgd(){
    python make_table.py --dataset celeba --paradigm signsgd
    python make_table.py --dataset cifar --paradigm signsgd
    python make_table.py --dataset cifar100 --paradigm signsgd
    python make_table.py --dataset fmnist --paradigm signsgd
    python make_table.py --dataset mnist --paradigm signsgd
    # python make_table.py --dataset imagenet --paradigm signsgd
}

signsgd_lbgm(){
    python make_table.py --dataset celeba --paradigm signsgd_lbgm
    python make_table.py --dataset cifar --paradigm signsgd_lbgm
    python make_table.py --dataset cifar100 --paradigm signsgd_lbgm
    python make_table.py --dataset fmnist --paradigm signsgd_lbgm
    python make_table.py --dataset mnist --paradigm signsgd_lbgm
    # python make_table.py --dataset imagenet --paradigm signsgd_lbgm
}

topk(){
    python make_table.py --dataset celeba --paradigm topk
    python make_table.py --dataset cifar --paradigm topk
    python make_table.py --dataset cifar100 --paradigm topk
    python make_table.py --dataset fmnist --paradigm topk
    python make_table.py --dataset mnist --paradigm topk
    # python make_table.py --dataset imagenet --paradigm topk
}

topk_lbgm(){
    python make_table.py --dataset celeba --paradigm topk_lbgm
    python make_table.py --dataset cifar --paradigm topk_lbgm
    python make_table.py --dataset cifar100 --paradigm topk_lbgm
    python make_table.py --dataset fmnist --paradigm topk_lbgm
    python make_table.py --dataset mnist --paradigm topk_lbgm
    # python make_table.py --dataset imagenet --paradigm topk_lbgm
}


$1
