################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 2 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 2 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 2 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 2 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 2 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
}



################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
}



################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
    python train_federated.py --device-id 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
}


$1
