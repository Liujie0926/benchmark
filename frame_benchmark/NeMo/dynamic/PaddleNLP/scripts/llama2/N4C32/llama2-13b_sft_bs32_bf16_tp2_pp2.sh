param="model_name_or_path=Llama-2-13b-hf "
param+="tensor_model_parallel_size=2 "
param+="pipeline_model_parallel_size=2 "
param+="sequence_parallel=True "
param+="recompute=null "
param+="activations_checkpoint_method=null "
param+="recompute_layers=null "
param+="scheme=none "
param+="bs_item=32 "
param+="fp_item=bf16 "
param+="run_stage=sft "
param+="run_mode=tp2_pp2 "
param+="device_num=N4C32 "
param+="model_item=llama2-13b_sft "

bash -c "${param} source prepare.sh";
bash -c "${param} bash run_benchmark.sh"