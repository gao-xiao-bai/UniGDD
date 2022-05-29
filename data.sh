python process_doc.py --output_dir . --cache_dir ./cache

python seq2seq_utils.py --split train --output_dir task_all --cache_dir ./cache --task 1,2,3
python seq2seq_utils.py --split validation --output_dir task_all --cache_dir ./cache --task 1,2,3

python seq2seq_utils.py --split train --output_dir task_main --cache_dir ./cache --task 1
python seq2seq_utils.py --split validation --output_dir task_main --cache_dir ./cache --task 1

