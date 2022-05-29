
CUDA_VISIBLE_DEVICES=2 \
python finetune_trainer.py \
  --data_dir ./task_main \
  --cache_dir ./cache \
  --output_dir ./output/5/eval\
  --num_train_epochs 5 \
  --model_name_or_path t5-base \
  --learning_rate 1e-4 \
  --adam_epsilon 1e-06 \
  --do_eval \
  --temp_start 0.8 \
  --temp_end 0.8 \
  --scheduler constant \
  --add_tokens \
  --eval_beams 2 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --overwrite_output_dir \
  --max_source_length 2560 \
  --max_target_length 500 \
  --task translation \
  --warmup_steps 500 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --logging_steps 200 \
  --predict_with_generate \
  --save_total_limit 3 \
  --generation_max_length 500 \
  --generation_num_beams 2 \
  --load_checkpoint_from ./output/5/ \

