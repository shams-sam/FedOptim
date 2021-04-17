debug(){
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 50 --dry-run 1
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --pca-var 0.95 --dry-run 1
}

mnist_var(){
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --pca-var 0.99 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --pca-var 0.99 --dry-run 0
}

mnist(){
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 1 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 2 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 3 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 5 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 10 --pca-var 0.95 --dry-run 0

    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 1 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 2 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 3 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 5 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 10 --pca-var 0.95 --dry-run 0
}


fmnist(){
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 1 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 2 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 3 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 5 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --ncomponent 10 --dry-run 0

    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 1 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 2 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 3 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 5 --dry-run 0
    python gen_pca_dirs.py --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --ncomponent 10 --dry-run 0

}


cifar(){
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --ncomponent 1 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --ncomponent 2 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --ncomponent 3 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --ncomponent 5 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --ncomponent 10 --pca-var 0.95 --dry-run 0

    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --ncomponent 1 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --ncomponent 2 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --ncomponent 3 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --ncomponent 5 --pca-var 0.95 --dry-run 0
    python gen_pca_dirs.py --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --ncomponent 10 --pca-var 0.95 --dry-run 0
}

$1
