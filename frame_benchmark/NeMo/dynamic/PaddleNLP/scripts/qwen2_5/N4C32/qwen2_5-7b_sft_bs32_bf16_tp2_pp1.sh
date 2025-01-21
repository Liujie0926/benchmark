param="model_name_or_path=Qwen2.5-7B "
param+="tensor_model_parallel_size=2 "
param+="pipeline_model_parallel_size=1 "
param+="sequence_parallel=True "
param+="recompute=full "
param+="activations_checkpoint_method=block "
param+="recompute_layers=64 "
param+="scheme=none "
param+="bs_item=32 "
param+="fp_item=bf16 "
param+="run_stage=sft "
param+="run_mode=tp2_pp1 "
param+="device_num=N4C32 "
param+="model_item=qwen2_5-7b_sft "

bash -c "${param} source prepare.sh";
bash -c "${param} bash run_benchmark.sh"
