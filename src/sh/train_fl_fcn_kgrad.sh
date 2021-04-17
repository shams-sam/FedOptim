debug(){
    python train_model.py --device-id 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 1
}

mnist(){
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.1 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.2 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.4 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.8 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 1.0 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.1 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.2 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.4 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 0.8 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --device-id 0 1 2 --dataset mnist --clf fcn --optim sgd --paradigm kgrad --kgrads 1 --error-tol 1.0 --residual 1 --num-workers 100 --epochs 51 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}


$1
