################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 0 --dataset cifar --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 --dataset cifar --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}


################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 0 --dataset fmnist --clf fcn --paradigm lbgm --error-tol 0.01 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 1
    # python train_federated.py --device-id 0 --dataset fmnist --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}

################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}



$1
