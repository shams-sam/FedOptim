mnist_10(){
    python init_data.py --dataset mnist --num-nodes 10 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset mnist --num-nodes 10 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

mnist_100(){
    python init_data.py --dataset mnist --num-nodes 100 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset mnist --num-nodes 100 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

fmnist_10(){
    python init_data.py --dataset fmnist --num-nodes 10 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset fmnist --num-nodes 10 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

fmnist_100(){
    python init_data.py --dataset fmnist --num-nodes 100 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset fmnist --num-nodes 100 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}


cifar_10(){
    python init_data.py --dataset cifar --num-nodes 10 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset cifar --num-nodes 10 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

cifar_100(){
    python init_data.py --dataset cifar --num-nodes 100 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset cifar --num-nodes 100 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}


$1
