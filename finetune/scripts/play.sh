data_path=$1
lr=3e-5

echo "The provided data_path is $data_path"

for seed in 42
do
    for data in H3
    do 
        python train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/EMP/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_EMP_${data}_seed${seed} \
            --model_max_length 128 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 500 \
            --output_dir output/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 500 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done
done