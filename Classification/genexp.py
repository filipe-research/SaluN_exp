#base = "cifar10"
base = "cifar100"
# datapath = "/home/pesquisador/pesquisa/datasets"
datapath = "/mnt/hd_pesquisa/pesquisa/datasets"
nr = 0.8
method = "baseline"


for run in range(1,6):
    seed = run*10
    save_dir = f"exp_{base}_nr{nr}_{method}_200ep_run{run}"
    command = f"python3 main_train.py --arch resnet18 --dataset {base} --lr 0.1 --epochs 200  --data {datapath} --save_dir {save_dir} --noise_rate {nr} --indexes_to_replace [] --train_seed {seed} --seed {seed} > {save_dir}.txt;"
    print(command)
    print("\n")