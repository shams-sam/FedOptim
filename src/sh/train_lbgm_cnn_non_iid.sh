################################################################################
# cifar
################################################################################
cifar(){
    # python train_federated.py --device-id 0 1 2 --dataset cifar --clf cnn --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    # python train_federated.py --device-id 0 1 2 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    # python train_federated.py --device-id 0 1 2 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    # python train_federated.py --device-id 0 1 2 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}

cifar_residual(){
    # python train_federated.py --device-id 2 1 0 --dataset cifar --clf cnn --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    # python train_federated.py --device-id 2 1 0 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    # python train_federated.py --device-id 2 1 0 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    # python train_federated.py --device-id 2 1 0 --dataset cifar --clf cnn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-1 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
}



################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}

fmnist_residual(){
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset fmnist --clf cnn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
}



################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf cnn --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --repeat 1 --dry-run 0 &
}

mnist_residual(){
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf cnn --paradigm lbgm --error-tol 1.0 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf cnn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 1 --residual 1 --repeat 1 --dry-run 0 &
}


$1