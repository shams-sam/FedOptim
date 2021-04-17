debug(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm dga --sdir-full 0 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 1
}

mnist(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm dga --sdir-full 0 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm dga --sdir-full 0 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
}

cifar(){
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm dga --sdir-full 0 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm dga --sdir-full 0 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
}

mnist_full(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm dga --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm dga --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
}

cifar_full(){
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm dga --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --paradigm dga --sdir-full 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --num-dga 1 --dga-bs 16 --dry-run 0 &
}


$1
