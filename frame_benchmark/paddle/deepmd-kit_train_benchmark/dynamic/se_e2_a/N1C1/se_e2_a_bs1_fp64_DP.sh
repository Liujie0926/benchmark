model_item=se_e2_a
bs_item=1
fp_item=fp64
run_process_type='SingleP'
run_mode=DP
device_num=N1C1
# copy files
\cp train_benchmark/dynamic/se_e2_a/benchmark_common/PrepareEnv.sh ./
\cp train_benchmark/dynamic/se_e2_a/benchmark_common/analysis_log.py ./
\cp train_benchmark/dynamic/se_e2_a/benchmark_common/run_benchmark.sh ./
\cp train_benchmark/dynamic/se_e2_a/N1C1/se_e2_a_bs1_fp64_DP.sh ./


sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;