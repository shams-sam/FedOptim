################################################################################
# celeba
################################################################################
voc(){
    python train_federated.py --device-id 0 1 2 --dataset voc --clf unet --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-3 --momentum 0.9 --non-iid 0 --repeat 1 --dry-run 0 &
}


n=$2

$1

