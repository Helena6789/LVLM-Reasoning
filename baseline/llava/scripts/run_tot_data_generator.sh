python baseline/llava/tot_data_generator.py --subset-size 2000 --seed 123 \
    --data-root ./dataset/SMART101-release-v1/SMART101-Data/ \
    --output-root ./baseline/llava/output/llava-v1.6-mistral-7b_gpt4o_tot_generate_data/ \
    --max-depth 10 --split-type instance