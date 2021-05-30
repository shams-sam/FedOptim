################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 0 1 2 --dataset celeba --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0 &
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid $n --repeat 1 --dry-run 0 --save-model 1 &
}

################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}

################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}

################################################################################
# mnist:
################################################################################
svhn(){
    python train_federated.py --device-id 1 2 0 --dataset svhn --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 0.5 --dry-run 0 --save-model 1 &
}

n=$2

$1
