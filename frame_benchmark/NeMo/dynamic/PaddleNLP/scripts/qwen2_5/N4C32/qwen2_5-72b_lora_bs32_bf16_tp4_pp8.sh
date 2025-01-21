param="model_name_or_path=Qwen2.5-72B "
param+="tensor_model_parallel_size=4 "
param+="pipeline_model_parallel_size=8 "
param+="sequence_parallel=False "
param+="recompute=null "
param+="activations_checkpoint_method=null "
param+="recompute_layers=null "
param+="scheme=lora "
param+="bs_item=32 "
param+="fp_item=bf16 "
param+="run_stage=lora "
param+="run_mode=tp4_pp8 "
param+="device_num=N4C32 "
param+="model_item=qwen2_5-72b_lora "

bash -c "${param} source prepare.sh";
bash -c "${param} bash run_benchmark.sh"
