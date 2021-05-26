################################################################################
# celeba
################################################################################
celeba(){
    # python train_federated.py --device-id 1 2 --dataset celeba --clf fcn --paradigm topk --topk 0.1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 --dataset celeba --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0 &
}

celeba_residual(){
    # python train_federated.py --device-id 1 2 --dataset celeba --clf fcn --paradigm topk --topk 0.1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --loss-type mse --non-iid 0 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 --dataset celeba --clf fcn --paradigm topk --topk 0.01 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --loss-type mse --non-iid 0 --residual 1 --repeat 1 --dry-run 0 &
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 10 --repeat 1 --dry-run 0 &
}

cifar_residual(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
}



################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
}

fmnist_residual(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
}


################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
}

mnist_residual(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf fcn --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-5 --momentum 0.9 --non-iid 10 --residual 1 --repeat 1 --dry-run 1
}


$1
