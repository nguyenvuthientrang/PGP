lab=42

# process inputs
mkdir -p ./logs/attack_target_${lab}/output_surrogate
mkdir -p ./logs/attack_target_${lab}/output_trigger
mkdir -p ./logs/attack_target_${lab}/output_victim

python main_surrogate.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs/attack_target_${lab}/output_surrogate --epochs 5 --no_pgp --num_tasks 1 --target_lab $lab
python main_trigger.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs/attack_target_${lab}/output_trigger --epochs 5 --no_pgp --num_tasks 1 --target_lab $lab --surrogate_path logs/attack_target_${lab}/output_surrogate/checkpoint/task1_checkpoint.pth
python main_victim.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs/attack_target_${lab}/output_victim --epochs 5 --no_pgp --num_tasks 10 --target_lab $lab --noise_path logs/attack_target_${lab}/output_trigger/checkpoint/best_noise.npy

