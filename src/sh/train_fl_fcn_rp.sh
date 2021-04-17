debug(){
    python train_model.py --device-id 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 10000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 1
}

mnist(){
    # train using normal sgd on iid data
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 & 
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 1000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 10000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 100000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &        
    
    # train using normal sgd on non-iid data
    python train_model.py --device-id 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 1000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 10000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset mnist --clf fcn --optim sgd --paradigm rp --ncomponent 100000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}

fmnist(){
    # train using normal sgd on iid data
    python train_model.py --device-id 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 & 
    python train_model.py --device-id 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm rp --ncomponent 1000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm rp --ncomponent 10000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset fmnist --clf fcn --optim sgd --paradigm rp --ncomponent 100000 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &        
    
    # train using normal sgd on non-iid data
    # python train_model.py --device-id 1 2 --dataset ffmnist --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    # python train_model.py --device-id 1 2 --dataset ffmnist --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}

cifar(){
    # train using normal sgd on iid data
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 & 
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 1000 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 10000 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 100000 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &        
    
    # train using normal sgd on non-iid data
    python train_model.py --device-id 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 100 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 1000 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 10000 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 1 2 --dataset cifar --clf fcn --optim sgd --paradigm rp --ncomponent 100000 --num-workers 100 --epochs 51 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
}

$1
