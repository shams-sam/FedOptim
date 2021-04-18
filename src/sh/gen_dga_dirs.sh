debug(){
    python gen_dga_dirs.py --device-id 0 --dataset mnist --clf resnet18 --optim sgd --num-workers 10 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --ncomponent 2 --dry-run 1
    # python gen_dga_dirs.py --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --pca-var 0.95 --dry-run 1
}

mnist(){
    python gen_dga_dirs.py --device-id 0 --dataset mnist --clf resnet18 --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dga-bs 16 --num-dga 1 --ncomponent 5 --dry-run 0
    python gen_dga_dirs.py --device-id 0 --dataset mnist --clf resnet18 --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --ncomponent 5 --dry-run 0

    python gen_dga_dirs.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dga-bs 16 --num-dga 1 --ncomponent 5 --dry-run 0
    python gen_dga_dirs.py --device-id 0 --dataset mnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --ncomponent 5 --dry-run 0
}

fmnist(){
    python gen_dga_dirs.py --device-id 1 --dataset fmnist --clf resnet18 --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0
    python gen_dga_dirs.py --device-id 1 --dataset fmnist --clf resnet18 --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0

    python gen_dga_dirs.py --device-id 1 --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0
    python gen_dga_dirs.py --device-id 1 --dataset fmnist --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0
    
}

cifar(){
    python gen_dga_dirs.py --device-id 1 --dataset cifar --clf resnet18 --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0
    python gen_dga_dirs.py --device-id 1 --dataset cifar --clf resnet18 --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0

    python gen_dga_dirs.py --device-id 1 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0
    python gen_dga_dirs.py --device-id 1 --dataset cifar --clf fcn --optim sgd --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dga-bs 16 --num-dga 1 --dry-run 0

}



$1
