mnist(){
    python init_models.py --dataset mnist --models fcn svm resnet18
}

fmnist(){
    python init_models.py --dataset fmnist --models fcn svm resnet18
}

cifar(){
    python init_models.py --dataset cifar --models fcn svm resnet18
}


$1
