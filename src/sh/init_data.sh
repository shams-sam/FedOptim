# first arg: dataset name
# second arg: number of splits

celeba(){
    # non iid 0 implies skip iid non-iid distributed splits
    # label splitting in regression task is not trivial 
    python init_data.py --dataset celeba --num-nodes $n --non-iid 0 --repeat 0.4 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

cifar(){
    python init_data.py --dataset cifar --num-nodes $n --non-iid $i --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

# skip-test if distributing over 100 nodes since each node will get only 1 class example in test set
cifar100(){
    python init_data.py --dataset cifar100 --num-nodes $n --non-iid $i --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --skip-test 0 --dry-run 0
}

fmnist(){
    python init_data.py --dataset fmnist --num-nodes $n --non-iid $i --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

# skip-test if distributing over 100 nodes since each node will get only 1 class example in test set
miniimagenet(){
    python init_data.py --dataset miniimagenet --num-nodes $n --non-iid $i --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --skip-test 1 --dry-run 0
}

mnist(){
    python init_data.py --dataset mnist --num-nodes $n --non-iid $i --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}

voc(){
    # non iid 0 implies skip iid non-iid distributed splits
    # label splitting in regression task is not trivial 
    python init_data.py --dataset voc --num-nodes $n --non-iid 0 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
}


n=$2
i=$3

$1
