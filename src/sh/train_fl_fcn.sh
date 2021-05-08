debug(){
    python train_model.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 10 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 1
}

mnist(){
    # train using normal sgd on iid data
    python train_model.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    # train using normal sgd on non-iid data
    python train_model.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}


fmnist(){
    # train using normal sgd on iid data
    python train_model.py --device-id 0 --dataset fmnist --clf fcn --optim sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    # train using normal sgd on non-iid data
    python train_model.py --device-id 0 --dataset fmnist --clf fcn --optim sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}


cifar(){
    # train using normal sgd on iid data
    python train_model.py --device-id 0 --dataset cifar --clf fcn --optim sgd --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    # train using normal sgd on non-iid data
    python train_model.py --device-id 0 --dataset cifar --clf fcn --optim sgd --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
}


$1
