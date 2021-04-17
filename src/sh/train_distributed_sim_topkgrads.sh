python train_distributed_sim_topkgrad_approx_grad.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --kgrads 10 --topk 78 --batch-size 0 --epochs 5 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_distributed_sim_topkgrad_approx_grad.py --device-id 1 --dataset mnist --clf fcn --paradigm sgd --kgrads 10 --topk 780 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &

# python train_distributed_sim_topkgrad_approx_grad.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --kgrads 10 --topk 78 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
# python train_distributed_sim_topkgrad_approx_grad.py --device-id 1 --dataset mnist --clf fcn --paradigm adam --kgrads 10 --topk 780 --batch-size 0 --epochs 101 --lr 0.01 --repeat 1 --dry-run 0 --early-stopping 0 &
