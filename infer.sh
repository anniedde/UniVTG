python3 -m main.inference_mr \
--dset_name scanqa \
--eval_split_name val \
--eval_path eval_transformed_full.jsonl \
--v_feat_dirs features/vid \
--t_feat_dir features/txt_transformed_full \
--t_feat_type features \
--resume results/omni/model_best.ckpt \
--ctx_mode video_tef