mnist(){
    # MNIST IID
    python init_data.py --dataset mnist --num-nodes 10 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset mnist --num-nodes 100 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0

    # MNIST NON-IID
    python init_data.py --dataset mnist --num-nodes 10 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset mnist --num-nodes 100 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

fmnist(){
    # FMNIST IID
    python init_data.py --dataset fmnist --num-nodes 10 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset fmnist --num-nodes 100 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0

    # FMNIST NON-IID
    python init_data.py --dataset fmnist --num-nodes 10 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset fmnist --num-nodes 100 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}


cifar(){
    # CIFAR IID
    python init_data.py --dataset cifar --num-nodes 10 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset cifar --num-nodes 100 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0

    # CIFAR NON-IID
    python init_data.py --dataset cifar --num-nodes 10 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset cifar --num-nodes 100 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}


$1
