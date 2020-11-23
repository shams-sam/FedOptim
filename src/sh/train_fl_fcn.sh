# training on 1 client network with iid dataset; same as the centralized training
iid_lr_01() {
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
}

# -------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_01() {
# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}


# training on 1 client network with iid dataset; same as the centralized training
iid_lr_01_noisy() {
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &

# training on 100 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 --noise 1 &
}

# -------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_01_noisy() {
# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &

# training on 100 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 --noise 1 &
}
# -------------------------------------------------

iid_lr_1() {
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset mnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
}

# -------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_1() {
# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset mnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
}

# training on 1 client network with iid dataset; same as the centralized training


iid_lr_01_fmnist() {
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm orth --num-workers 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
}

# -------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_01_fmnist() {
# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
}
# -------------------------------------------------

iid_lr_1_fmnist() {
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm sgd --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm adam --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm orth --num-workers 1 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 10 client network with iid dataset
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --device-id 1 --dataset fmnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.1 --non-iid 10 --repeat 1 --dry-run 0 &
}

# -------------------------------------------------


# training on 10 client network with iid dataset
non_iid_lr_1_fmnist() {
# training on 10 client network with iid dataset
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm sgd --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm adam --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm orth --num-workers 10 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &

# training on 100 client network with iid dataset
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm sgd --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm adam --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev pr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm conj --conj-dev fr --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
python train_model.py --device-id 2 --dataset fmnist --clf fcn --paradigm orth --num-workers 100 --epochs 101 --lr 0.1 --non-iid 1 --repeat 1 --dry-run 0 &
}


$1
