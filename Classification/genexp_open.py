# base = "cifar10"
# base = "cifar100"
# base = "cifar10_idn"
# base = "cifar100_idn"
base = "cifar10_open"



# datapath = "/home/pesquisador/pesquisa/datasets"
datapath = "/mnt/hd_pesquisa/pesquisa/datasets"


###
### No codigo em dataset.py, está assim:
#### num_total_noise = int(noise_rate * len(train_idx))
#### num_open_noise = int(open_ratio * num_total_noise)


##### closed / open  (segundo artigo)
parameter_closed = 0.0
parameter_open = 0.6
nr = parameter_closed + parameter_open
open_r = parameter_open/nr


method = "baseline"
# method = "retrain"
# method = "salun"

# noise_mode = "sym"
# noise_mode = "asym"

#python3 main_train.py --arch resnet18 --dataset cifar10_open --epochs 300 --noise_rate 0.3 --open_ratio 0.15 --data /mnt/hd_pesquisa/pesquisa/datasets --save_dir exp_cifar10_open_0.15_0.15


for run in range(1,6):
    seed = run*10
    save_dir = f"exp_{base}_closed{parameter_closed}_open{parameter_open}_{method}_200ep_run{run}"

    if method == "baseline":
        command = f"python3 main_train.py --arch resnet18 --dataset {base} --lr 0.1 --epochs 200  --data {datapath} --save_dir {save_dir} --noise_rate {nr} --open_ratio {open_r} --indexes_to_replace [] --train_seed {seed} --seed {seed}  > {save_dir}.txt;"
    elif method == "retrain":
        model_path = f"exp_{base}_nr{nr}_baseline_200ep_run{run}"
        command = f"python3 main_forget.py --dataset {base} --unlearn_epochs 200 --noise_rate {nr} --open_ratio {open_r} --data {datapath} --save_dir {save_dir} --indexes_to_replace [] --unlearn retrain --unlearn_lr 0.1 --model_path {model_path}/0model_SA_best.pth.tar  --train_seed {seed} --seed {seed} > {save_dir}.txt;"
    print(command)
    print("\n")