debug(){
    # python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 1
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --sdir-full 0 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 1
}

mnist_var(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --pca-var 0.95 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --pca-var 0.99 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --pca-var 0.95 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --pca-var 0.99 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}

mnist(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 2 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 3 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 5 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 10 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 2 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 3 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 5 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 10 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}


mnist_sdir_full(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 2 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 3 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 5 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 10 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 2 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 3 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 5 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm pca --ncomponent 10 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}


fmnist(){
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 2 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 3 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 5 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 10 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 2 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 3 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 5 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm pca --ncomponent 10 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}

cifar(){
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 2 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 3 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 5 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 10 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 2 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 3 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 5 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 10 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
}


cifar_sdir_full(){
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 1 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 2 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 3 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 5 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 10 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 1 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 2 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 3 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 5 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm pca --ncomponent 10 --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
}


$1
