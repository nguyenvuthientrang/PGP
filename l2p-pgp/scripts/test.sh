lab=2

# process inputs
# mkdir -p ./logs_simulate/attack_target_${lab}/output_surrogate
mkdir -p ./test/attack_target_${lab}/output_trigger
# mkdir -p ./logs_simulate/attack_target_${lab}/output_simulate
# mkdir -p ./logs_simulate/attack_target_${lab}/output_victim

# python main_surrogate.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs_simulate/attack_target_${lab}/output_surrogate --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab
python main_trigger.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./test/attack_target_${lab}/output_trigger --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab --surrogate_path test/surrogate/task1_checkpoint.pth
# python main_simulate.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs_simulate/attack_target_${lab}/output_simulate --epochs 5 --no_pgp --num_tasks 2 --target_lab $lab --noise_path logs_simulate/attack_target_2/output_trigger/checkpoint/best_noise.npy
# python main_victim.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./logs_simulate/attack_target_${lab}/output_victim --epochs 5 --no_pgp --num_tasks 10 --target_lab $lab --noise_path logs_simulate/attack_target_2/output_simulate/checkpoint/best_noise_tuned.npy