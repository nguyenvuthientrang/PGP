lab=2


# process inputs
# mkdir -p ./l2p-pgp/output_test_adversarial_noise

# python main_victim.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir output_non_attack \
#         --num_tasks 10 --no_pgp --target_lab $lab --noise_path output_non_attack/trigger/best_noise_tuned.npy --eval

# python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir output_non_attack_cub200 \
#         --num_tasks 5 --no_pgp --target_lab $lab --noise_path output_non_attack_cub200/simulation/best_noise_tuned.npy --eval

# mkdir -p ./output_attack_cub200

python main_victim.py cub200_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir output_non_attack_cub200 \
        --num_tasks 5 --no_pgp --target_lab $lab --noise_path logs_bce_cosine_sum/cub200/attack_target_2/output_simulate_noise_task/tri_lr_1e-2_p_lr_0.03_tri_epochs_50/tri_reg_coef_1e-2/checkpoint/best_noise_tuned.npy --eval

# python main_victim.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --epochs 5 --output_dir output_non_attack \
#         --num_tasks 10 --no_pgp --target_lab $lab --noise_path logs_bce/cifar100/attack_target_2/output_simulate_noise_task/tri_lr_1e-2_p_lr_0.03_tri_epochs_20/checkpoint/best_noise_tuned.npy --eval
