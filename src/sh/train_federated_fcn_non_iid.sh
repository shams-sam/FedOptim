################################################################################
# cifar: to-do
################################################################################
cifar(){
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 1 --repeat 1 --dry-run 0 --save-model 1 &
}


################################################################################
# fmnist: done
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}

################################################################################
# mnist: done
################################################################################
mnist(){
    python train_federated.py --device-id 1 0 2 --dataset mnist --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}



$1
