debug(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 16 --non-iid 10 --repeat 1 --dry-run 1
}

mnist_iid(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 512  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 256  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 128  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 64  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 32  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 16  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 8  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 4  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 2  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 1  --non-iid 10 --repeat 1 --dry-run 0 &
}

mnist_non_iid(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 512  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 256  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 128  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 64  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 32  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 16  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 8  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 4  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 2  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --batch-size 1  --non-iid 1 --repeat 1 --dry-run 0 &
}

cifar_iid(){
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 512  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 256  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 128  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 64  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 32  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 16  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 8  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 4  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 2  --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 1  --non-iid 10 --repeat 1 --dry-run 0 &
}

cifar_non_iid(){
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 512  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 256  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 128  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 64  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 32  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 16  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 8  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 4  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 2  --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.1 --batch-size 1  --non-iid 1 --repeat 1 --dry-run 0 &
}


$1
