mnist(){
    python init_data.py --dataset mnist --num-nodes $n --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset mnist --num-nodes $n --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

fmnist(){
    python init_data.py --dataset fmnist --num-nodes $n --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset fmnist --num-nodes $n --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

cifar(){
    python init_data.py --dataset cifar --num-nodes $n --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
    python init_data.py --dataset cifar --num-nodes $n --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

n=$2

$1
