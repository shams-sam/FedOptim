################################################################################
# celeba: done
################################################################################
voc(){
    python train_federated.py --device-id 1 0 2 --dataset voc --clf unet --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-3 --momentum 0.9 --non-iid 0 --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 0 2 --dataset voc --clf unet --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-3 --momentum 0.9 --non-iid 0 --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset voc --clf unet --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-3 --momentum 0.9 --non-iid 0 --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset voc --clf unet --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-3 --momentum 0.9 --non-iid 0 --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 2 1 --dataset voc --clf unet --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-3 --momentum 0.9 --non-iid 0 --residual 0 --repeat 1 --dry-run 0 &
}

$1
