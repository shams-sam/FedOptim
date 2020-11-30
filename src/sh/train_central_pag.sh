# python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-comp 1 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-comp 5 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-comp 101 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-comp 1 --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-comp 5 --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --num-comp 101 --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &


python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-comp 1 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-comp 5 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-comp 101 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-comp 1 --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-comp 5 --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized_pca_approx_grad.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --num-comp 101 --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
