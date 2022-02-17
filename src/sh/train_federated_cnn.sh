################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 0 1 2 --dataset celeba --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-5 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0 &
}

celeba_sampled(){
    python train_federated.py --device-id 0 1 2 --dataset celeba --clf cnn --optim sgd --num-workers 100 --sample 1 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-5 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0 &
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9  --non-iid $n --repeat 1 --dry-run 0 &
}

cifar_sampled(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf cnn --optim sgd --num-workers 100 --sample 1 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9  --non-iid $n --repeat 1 --dry-run 0 &
}


################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}

fmnist_sampled(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --optim sgd --num-workers 100 --sample 0.5 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --optim sgd --num-workers 100 --sample 0.3 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --optim sgd --num-workers 100 --sample 0.1 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &

}

################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf cnn --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}

mnist_sampled(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf cnn --optim sgd --num-workers 100 --sample 0.3 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf cnn --optim sgd --num-workers 100 --sample 0.1 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf cnn --optim sgd --num-workers 100 --sample 0.01 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}


n=$2

$1
