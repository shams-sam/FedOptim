ckpt_dir = '/root/WorkSpace/FedOptim/ckpts'
data_dir = '/root/WorkSpace/data'
tb_dir = '/root/WorkSpace/FedOptim/runs'
download = False

num_trains = {
    'celeba': 81385, # total 162770
    'cifar': 50000,
    'coco': 59144, # total 118287
    'fmnist': 60000,
    'mnist': 60000,
}

num_tests = {
    'celeba': 9934, # total 19867
    'cifar': 10000,
    'coco': 5000, # total 5000
    'fmnist': 10000,
    'mnist': 10000,
}

input_sizes = {
    'celeba': 3*32*32,
    'cifar': 3*32*32,
    'coco': 3*32*32,
    'fmnist': 28*28,
    'mnist': 28*28,
}

output_sizes = {
    'celeba': 10,
    'cifar': 10,
    'coco': 32*32,
    'fmnist': 10,    
    'mnist': 10,
}

num_channels = {
    'celeba': 3,
    'cifar': 3,
    'coco': 3,
    'fmnist': 1,
    'mnist': 1,
}

cnn_view = {
    'celeba': 5*5*50,
    'cifar': 5*5*50,
    'fmnist': 4*4*50,
    'mnist': 4*4*50,
}

im_size = {
    'celeba': 32,
    'coco': 32,
}
