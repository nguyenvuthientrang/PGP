lab=2

for tri_lr in 1e-2 1e-3
do
for p_lr in 1e-3 0.03 1e-4
do
for tri_epochs in 20 #10
do
# process inputs
mkdir -p ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/

python main_simulate.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs} \
        --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab \
        --surrogate_path logs_simulate/attack_target_2/output_surrogate/checkpoint/task1_checkpoint.pth \
        --noise_path logs_simulate/attack_target_2/output_trigger/checkpoint/best_noise.npy \
        --simulate_round_tri $tri_epochs --simulate_round_prompt 1 \
        --simulate_lr_prompt $p_lr --simulate_lr_tri $tri_lr
done
done
done
