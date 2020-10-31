# training on 1 client network with iid dataset; same as the centralized training
iid_lr_01() {
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm conj --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm conj --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 500 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 500 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 500 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
}

# -------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_01() {
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 0 --dataset mnist --clf fcn --paradigm conj --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 500 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 500 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 500 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}

# -------------------------------------------------


# training on 1 client network with iid dataset; same as the centralized training
iid_lr_1() {
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 1 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 10 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 100 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 500 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 500 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --num-workers 500 --epochs 101 --lr 1.0 --non-iid 10 --repeat 1 --dry-run 0 &
}

# ------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_1() {
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --num-workers 10 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --num-workers 100 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 500 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 500 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --num-workers 500 --epochs 101 --lr 1.0 --non-iid 1 --repeat 1 --dry-run 0 &
}

$1
