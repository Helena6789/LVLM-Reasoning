python ../baseline.py --puzzle-max 101 --subset-size 1000 --seed 123 \
    --data-root /home/hq/LVLM/LVLM-Reasoning/dataset/SMART101-release-v1/SMART101-Data/ \
    --smart-info-v2-csv /home/hq/LVLM/LVLM-Reasoning/dataset/SMART_info_v2.csv \
    --model-id /home/hq/models/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f \
    --output-root /home/hq/LVLM/LVLM-Reasoning/baseline/llava/output/llava-v1.6-mistral-7b_baseline_original_data_output2 \