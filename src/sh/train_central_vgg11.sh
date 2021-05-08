################################################################################
# cifar
################################################################################
# python train_centralized.py --device-id 2 1 0 --dataset cifar --clf vgg11 --paradigm sgd --batch-size 25000 --epochs 151 --lr 1e-2 --momentum 0.9 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0
python train_centralized.py --device-id 1 2 0 --dataset cifar --clf vgg11 --paradigm sgd --batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0
