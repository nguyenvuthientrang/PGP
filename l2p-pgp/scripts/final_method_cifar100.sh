lab=2

OUTDIR=logs_simulate/cifar100/attack_target_${lab}

# # # process inputs
# mkdir -p ./${OUTDIR}/output_surrogate
# mkdir -p ./${OUTDIR}/output_trigger

# python main_surrogate.py cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./${OUTDIR}/output_surrogate --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab
# python main_trigger.py cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./${OUTDIR}/output_trigger --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab --surrogate_path ${OUTDIR}/output_surrogate/checkpoint/task1_checkpoint.pth


for tri_lr in 1e-2 #1e-3
do
for p_lr in 0.03 #1e-3 1e-4 #0.03 #1e-3 0.03 1e-4
do
for tri_reg_coef in 1e-2 #1e-3 1e-1 #1e-4 1e-3 1e-1 1e-5
do
for tri_epochs in 50 20 #10 #20
do
# process inputs
SUBOUTDIR=logs_bce_cosine_sum/cifar100/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/tri_reg_coef_${tri_reg_coef}
# mkdir -p ./${SUBOUTDIR}

# python main_simulate.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./${SUBOUTDIR} \
#         --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab \
#         --surrogate2_path logs_simulate/attack_target_2/output_surrogate/checkpoint/task2_checkpoint.pth \
#         --noise_path logs_simulate/attack_target_2/output_trigger/checkpoint/best_noise.npy \
#         --simulate_round_tri $tri_epochs --simulate_round_prompt 1 \
#         --simulate_lr_prompt $p_lr --simulate_lr_tri $tri_lr --use_bce --push --tri_reg_distance cosine --tri_reg_reduction sum --tri_reg_coef $tri_reg_coef

# mkdir -p ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/output_victim_pgp
# mkdir -p ./${SUBOUTDIR}/output_victim

# # python main_victim.py cifar100_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/cifar100/output_victim_pgp \
# #         --num_tasks 10  --target_lab $lab --noise_path logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/checkpoint/best_noise_tuned.npy

# python main_victim.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir .${SUBOUTDIR}/output_victim \
#         --num_tasks 10 --no_pgp --target_lab $lab --noise_path ${SUBOUTDIR}/checkpoint/best_noise_tuned.npy  --use_bce --push --tri_reg_distance cosine --tri_reg_reduction sum --tri_reg_coef $tri_reg_coef


mkdir -p ./${SUBOUTDIR}/output_victim_pgp

# python main_victim.py cifar100_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/cifar100/output_victim_pgp \
#         --num_tasks 10  --target_lab $lab --noise_path logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/checkpoint/best_noise_tuned.npy

python main_victim.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir .${SUBOUTDIR}/output_victim_pgp \
        --num_tasks 10 --target_lab $lab --noise_path ${SUBOUTDIR}/checkpoint/best_noise_tuned.npy  --use_bce --push --tri_reg_distance cosine --tri_reg_reduction sum --tri_reg_coef $tri_reg_coef

done
done
done
done