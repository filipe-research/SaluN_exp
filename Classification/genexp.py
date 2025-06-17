# base = "cifar10"
# base = "cifar100"
base = "cifar10_idn"
# base = "cifar100_idn"


datapath = "/home/pesquisador/pesquisa/datasets"
# datapath = "/mnt/hd_pesquisa/pesquisa/datasets"

nr = 0.2

method = "baseline"
# method = "retrain"
# method = "salun"

noise_mode = "sym"
# noise_mode = "asym"


for run in range(1,6):
    seed = run*10
    save_dir = f"exp_{base}_nr{nr}_{method}_200ep_run{run}"

    if method == "baseline":
        command = f"python3 main_train.py --arch resnet18 --dataset {base} --lr 0.1 --epochs 200  --data {datapath} --save_dir {save_dir} --noise_rate {nr} --indexes_to_replace [] --train_seed {seed} --seed {seed} --noise_mode {noise_mode} > {save_dir}.txt;"
    elif method == "retrain":
        model_path = f"exp_{base}_nr{nr}_baseline_200ep_run{run}"
        command = f"python3 main_forget.py --dataset {base} --unlearn_epochs 200 --noise_rate {nr} --data {datapath} --save_dir {save_dir} --indexes_to_replace [] --unlearn retrain --unlearn_lr 0.1 --model_path {model_path}/0model_SA_best.pth.tar ----noise_mode {noise_mode} --train_seed {seed} --seed {seed} > {save_dir}.txt;"
    print(command)
    print("\n")