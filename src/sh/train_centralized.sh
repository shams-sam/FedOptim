vanilla(){
    python train_centralized.py --device-id 1 2 --dataset mnist --clf resnet18 --paradigm sgd --batch-size 128 --epochs 51 --lr 1e-2 --repeat 1 --dry-run 0 --early-stopping 0
}

rp(){
    # python train_centralized.py --device-id 1 2 --dataset mnist --clf resnet18 --paradigm sgd rp --ncomponent 10 --batch-size 128 --epochs 51 --lr 1e-2 --repeat 1 --dry-run 0 --early-stopping 0
    python train_centralized.py --device-id 2 1 --dataset mnist --clf resnet18 --paradigm sgd rp --ncomponent 20 --batch-size 128 --epochs 51 --lr 1e-2 --repeat 1 --dry-run 0 --early-stopping 0
}



$1
