lab=2

OUTDIR=logs_simulate/cub200/attack_target_${lab}

# # # process inputs
# mkdir -p ./${OUTDIR}/output_surrogate
# mkdir -p ./${OUTDIR}/output_trigger

# python main_surrogate.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${OUTDIR}/output_surrogate --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab
# python main_trigger.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${OUTDIR}/output_trigger --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab --surrogate_path ${OUTDIR}/output_surrogate/checkpoint/task1_checkpoint.pth


for tri_lr in 1e-2 #1e-2
do
for p_lr in 1e-3 #0.03 1e-4  #0.03 #1e-3 0.03 1e-4
do
for tri_epochs in 50 #10 20
do
for cont_mask_type in add mul
do
# process inputs
SUBOUTDIR=logs_continuous_mask/cub200/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/mask_type_${cont_mask_type}
mkdir -p ./${SUBOUTDIR}

python main_simulate.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${SUBOUTDIR} \
        --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab \
        --surrogate2_path ${OUTDIR}/output_surrogate/checkpoint/task2_checkpoint.pth \
        --noise_path ${OUTDIR}/output_trigger/checkpoint/best_noise.npy \
        --simulate_round_tri $tri_epochs --simulate_round_prompt 1 \
        --simulate_lr_prompt $p_lr --simulate_lr_tri $tri_lr \
        --continuous_mask --cont_mask_type $cont_mask_type

# mkdir -p ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/output_victim_pgp
mkdir -p ./${SUBOUTDIR}/output_victim

# python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/cub200/output_victim_pgp \
#         --num_tasks 10  --target_lab $lab --noise_path logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/checkpoint/best_noise_tuned.npy

python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir .${SUBOUTDIR}/output_victim \
        --num_tasks 5 --no_pgp --target_lab $lab --noise_path ${SUBOUTDIR}/checkpoint/best_noise_tuned.npy \
        --continuous_mask --cont_mask_type $cont_mask_type

done
done
done
done

for tri_lr in 1e-2 #1e-2
do
for p_lr in 1e-3 0.03 1e-4  #0.03 #1e-3 0.03 1e-4
do
for tri_epochs in 100 #10 20
do
for cont_mask_type in add mul
do
# process inputs
SUBOUTDIR=logs_continuous_mask/cub200/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/mask_type_${cont_mask_type}
mkdir -p ./${SUBOUTDIR}

python main_simulate.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${SUBOUTDIR} \
        --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab \
        --surrogate2_path ${OUTDIR}/output_surrogate/checkpoint/task2_checkpoint.pth \
        --noise_path ${OUTDIR}/output_trigger/checkpoint/best_noise.npy \
        --simulate_round_tri $tri_epochs --simulate_round_prompt 1 \
        --simulate_lr_prompt $p_lr --simulate_lr_tri $tri_lr \
        --continuous_mask --cont_mask_type $cont_mask_type

# mkdir -p ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/output_victim_pgp
mkdir -p ./${SUBOUTDIR}/output_victim

# python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/cub200/output_victim_pgp \
#         --num_tasks 10  --target_lab $lab --noise_path logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/checkpoint/best_noise_tuned.npy

python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir .${SUBOUTDIR}/output_victim \
        --num_tasks 5 --no_pgp --target_lab $lab --noise_path ${SUBOUTDIR}/checkpoint/best_noise_tuned.npy \
        --continuous_mask --cont_mask_type $cont_mask_type --tuning

done
done
done
done