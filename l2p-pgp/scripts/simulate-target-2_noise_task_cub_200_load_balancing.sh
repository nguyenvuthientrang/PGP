lab=2

OUTDIR=logs_simulate/cub200/attack_target_${lab}


# # # process inputs
# mkdir -p ./${OUTDIR}/output_surrogate
# mkdir -p ./${OUTDIR}/output_trigger

for load_balancing_coef in 1e-2 1e-3 1e-4 1e-1
do
VIC_OUTDIR=logs_load_balancing/cub200/attack_target_${lab}/load_balancing_coef_${load_balancing_coef}
mkdir -p ./${VIC_OUTDIR}/output_trigger
# python main_surrogate.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${OUTDIR}/output_surrogate --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab
python main_trigger.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${VIC_OUTDIR}/output_trigger --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab --surrogate_path ${OUTDIR}/output_surrogate/checkpoint/task1_checkpoint.pth \
        --load_balancing --load_balancing_coef $load_balancing_coef 
done

for tri_lr in 1e-2 1e-3
do
for p_lr in 0.03 1e-3 1e-4 #0.03 #1e-3 0.03 1e-4
do
for tri_epochs in 20 #10 20
do
for load_balancing_coef in 1e-2 1e-3 1e-4 1e-1
do
VIC_OUTDIR=logs_load_balancing/cub200/attack_target_${lab}/load_balancing_coef_${load_balancing_coef}
# process inputs
SUBOUTDIR=logs_load_balancing/cub200/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/load_balancing_coef_${load_balancing_coef}
mkdir -p ./${SUBOUTDIR}

python main_simulate.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./${SUBOUTDIR} \
        --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab \
        --surrogate2_path ${OUTDIR}/output_surrogate/checkpoint/task2_checkpoint.pth \
        --noise_path ${VIC_OUTDIR}/output_trigger/checkpoint/best_noise.npy \
        --simulate_round_tri $tri_epochs --simulate_round_prompt 1 \
        --simulate_lr_prompt $p_lr --simulate_lr_tri $tri_lr 

# mkdir -p ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/output_victim_pgp
mkdir -p ./${SUBOUTDIR}/output_victim

# python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/cub200/output_victim_pgp \
#         --num_tasks 10  --target_lab $lab --noise_path logs_simulate/attack_target_${lab}/output_simulate_noise_task/tri_lr_${tri_lr}_p_lr_${p_lr}_tri_epochs_${tri_epochs}/checkpoint/best_noise_tuned.npy

python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir ${SUBOUTDIR}/output_victim \
        --num_tasks 5 --no_pgp --target_lab $lab --noise_path ${SUBOUTDIR}/checkpoint/best_noise_tuned.npy --tuning

done
done
done
done
