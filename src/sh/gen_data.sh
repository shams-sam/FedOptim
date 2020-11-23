# MNIST IID
python gen_data.py --dataset mnist --num-nodes 1 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset mnist --num-nodes 10 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset mnist --num-nodes 100 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset mnist --num-nodes 500 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0


# MNIST NON-IID
python gen_data.py --dataset mnist --num-nodes 10 \
       --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset mnist --num-nodes 100 \
       --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset mnist --num-nodes 500 \
       --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0


# FMNIST IID
python gen_data.py --dataset fmnist --num-nodes 1 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset fmnist --num-nodes 10 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset fmnist --num-nodes 100 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset fmnist --num-nodes 500 \
       --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0


# FMNIST NON-IID
python gen_data.py --dataset fmnist --num-nodes 10 \
       --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset fmnist --num-nodes 100 \
       --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0
python gen_data.py --dataset fmnist --num-nodes 500 \
       --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform 1 --dry-run 0

