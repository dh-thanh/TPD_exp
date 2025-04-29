# VITONHD_release_person_combine_garment_240epochs
CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
--logdir train_logs/VITONHD/ \
--pretrained_model /workspace/Try-on-Product/huggingface_models/TPD-checkpoints/release/TPD_240epochs.ckpt \
--base configs/train/train_VITONHD.yaml \
--scale_lr False \
--name flowfield_loss_for_decoder \
--logdir /HDD/Projects/Try-on-Product/train_dir/TPD/
# --resume_from_checkpoint /workspace/Try-on-Product/huggingface_models/TPD-checkpoints/release/TPD_240epochs.ckpt