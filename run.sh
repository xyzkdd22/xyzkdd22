# command to run the semi_self_supervised framework
python moco_main.py \
--arch 'r2plus1d_18' \
--batch_size 288 \
--start_epoch 0 \
--num_labeled 400 \
--threshold 0.95 \
--epochs 600 \
--gpus 0 1 2 3 4 5 \
--eval_step 33 \
--mu 2.75 \
--beta 1 \
--num_classes 101 \
--num_frames 16 \
--frame_size 112 \
--results_dir "results" \
--npy_train_dir "./ucf10/training_npy/masked_vs_bp_v5_vs_mixed_targets" \
--semi_train_path "./ucf10/training_npy/semi_labeled_clip_vs_bp_v5_mixed_targets" \
--npy_root_dir_valid "./ucf10/training_npy/labeled_for_validation" 