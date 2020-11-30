# python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --batch-size 512 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --batch-size 2048 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 & 
# python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --batch-size 8192 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm sgd --batch-size 32768 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &

python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --batch-size 128 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --batch-size 512 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --batch-size 2048 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 & 
python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --batch-size 8192 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
python train_centralized.py --device-id 2 --dataset mnist --clf fcn --paradigm adam --batch-size 32768 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
