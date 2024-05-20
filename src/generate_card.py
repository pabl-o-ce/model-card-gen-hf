import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template

# Set up Jinja2 environment
current_file_path = Path(__file__).resolve()  # Resolve to get the absolute path
directory_path = current_file_path.parent  # Get the directory
template_dir = os.path.join(directory_path, 'templates')
print(template_dir)
env = Environment(loader=FileSystemLoader(template_dir))

# Load the template from a file
template = env.get_template('cognitive_computation.jinja')

# Define the model details
model_details = {
    'model_name': 'Dolphin 2.9.1 Qwen 110b',
    'model_tags': ['generated_from_trainer', 'axolotl', 'text-generation', 'autotrain_compatible', 'endpoints_compatible', 'chatml', 'text-generation-inference', 'transformers'],
    'model_base': ['Qwen/Qwen1.5-110B'],
    'model_licence': 'other',
    'model_licence_name': 'tongyi-qianwen',
    'model_licence_link': 'https://huggingface.co/Qwen/Qwen1.5-110B/blob/main/LICENSE',
    'languages': ['en'],
    'datasets': [
        'cognitivecomputations/Dolphin-2.9',
        'teknium/OpenHermes-2.5',
        'm-a-p/CodeFeedback-Filtered-Instruction',
        'cognitivecomputations/dolphin-coder',
        'cognitivecomputations/samantha-data',
        'microsoft/orca-math-word-problems-200k',
        'Locutusque/function-calling-chatml',
        'internlm/Agent-FLAN',
    ],
    'model_image': 'https://cdn-uploads.huggingface.co/production/uploads/63111b2d88942700629f5771/ldkN1J0WIDQwU4vutGYiD.png',
    'model_description': 'Curated and trained by Eric Hartford, Lucas Atkins, and Fernando Fernandes, and Cognitive Computations\nThis model is based on Qwen1.5-110B, and is governed by [tongyi-qianwen license](LICENSE)\nThe base model has 32k context, and the full-weight fine-tuning was with 8k sequence length.\nThis model was trained FFT on parameters selected by [Laser Scanner](https://github.com/cognitivecomputations/laserRMT/blob/main/laser_scanner.py), using ChatML prompt template format.',
    'sponsor_info': '- [Crusoe Cloud](https://crusoe.ai/) - provided excellent on-demand 8xH100 node',
    'model_repo': 'cognitivecomputations/dolphin-2.9.1-qwen-110b',
    'chat_template_name': 'chatml',
    'chat_template_format': '<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{asistant}<|im_end|>',
    'intended_use': 'Text generation and completion.',
    'limitations': 'May generate biased or offensive content.',
    'ethical_considerations': 'Use with caution and review generated content.',
    'model_author_gguf': '[@crusoeai](https://huggingface.co/crusoeai)',
    'model_link_gguf': 'https://huggingface.co/crusoeai/dolphin-2.9.1-qwen-110b-GGUF',
    'model_author_mlx': '[@mlx-community](https://huggingface.co/mlx-community)',
    'model_link_mlx': [
        'https://huggingface.co/mlx-community/dolphin-2.9.1-qwen-110b-2bit',
        'https://huggingface.co/mlx-community/dolphin-2.9.1-qwen-110b-4bit',
        'https://huggingface.co/mlx-community/dolphin-2.9.1-qwen-110b-8bit',
    ],
    'evals_img': 'https://cdn-uploads.huggingface.co/production/uploads/63111b2d88942700629f5771/U86Zu-MzLq4rECJRAAvgq.png',
    'framework_badge': '[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)',
    'axolotl_version': '0.4.0',
    'axolotl_config': '''base_model: /workspace/axolotl/qwen-checkpoint
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

# trust_remote_code: true

# load_in_8bit: true
# load_in_4bit: true
# strict: false

datasets:
  - path: /workspace/datasets/dolphin-2.9/dolphin201-sharegpt2.jsonl
    type: sharegpt
    conversation: chatml
  # - path: /workspace/datasets/dolphin-2.9/Ultrachat200kunfiltered.jsonl
  #   type: sharegpt
  #   conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/dolphin-coder-translate-sharegpt2.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/dolphin-coder-codegen-sharegpt2.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/m-a-p_Code-Feedback-sharegpt-unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/m-a-p_CodeFeedback-Filtered-Instruction-sharegpt-unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/not_samantha_norefusals.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/Orca-Math-resort-unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/agent_instruct_react_unfiltered.jsonl
    type: sharegpt  
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/toolbench_instruct_j1s1_3k_unfiltered.jsonl
    type: sharegpt  
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/toolbench_negative_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/toolbench_react_10p_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/toolbench_tflan_cot_30p_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9/openhermes200k_unfiltered.jsonl
    type: sharegpt 
    conversation: chatml
  # - path: /workspace/datasets/dolphin-2.9/SystemConversations.jsonl
  #   type: sharegpt
  #   conversation: chatml

chat_template: chatml
dataset_prepared_path: last_run_prepared
val_set_size: 0.01
output_dir: ./qwen-out

# adapter: qlora
# lora_r: 16
# lora_alpha: 16
# lora_modules_to_save: [embed_tokens, lm_head]
# lora_dropout: 0.05
# lora_target_linear: false

unfrozen_parameters:
- ^lm_head.weight$
- ^model.embed_tokens.weight$
# input_layernorm layers
- model.layers.0.input_layernorm
- model.layers.1.input_layernorm
- model.layers.2.input_layernorm
- model.layers.3.input_layernorm
- model.layers.4.input_layernorm
- model.layers.5.input_layernorm
- model.layers.6.input_layernorm
- model.layers.7.input_layernorm
- model.layers.8.input_layernorm
- model.layers.9.input_layernorm
- model.layers.10.input_layernorm
- model.layers.11.input_layernorm
- model.layers.12.input_layernorm
- model.layers.13.input_layernorm
- model.layers.14.input_layernorm
- model.layers.15.input_layernorm
- model.layers.16.input_layernorm
- model.layers.17.input_layernorm
- model.layers.18.input_layernorm
- model.layers.19.input_layernorm
- model.layers.20.input_layernorm
- model.layers.21.input_layernorm
- model.layers.22.input_layernorm
- model.layers.23.input_layernorm
# lm_head layers
# mlp.down_proj layers
- model.layers.17.mlp.down_proj
- model.layers.18.mlp.down_proj
- model.layers.19.mlp.down_proj
- model.layers.20.mlp.down_proj
- model.layers.21.mlp.down_proj
- model.layers.22.mlp.down_proj
- model.layers.23.mlp.down_proj
- model.layers.24.mlp.down_proj
- model.layers.25.mlp.down_proj
- model.layers.26.mlp.down_proj
- model.layers.27.mlp.down_proj
- model.layers.28.mlp.down_proj
- model.layers.29.mlp.down_proj
- model.layers.30.mlp.down_proj
- model.layers.31.mlp.down_proj
- model.layers.32.mlp.down_proj
- model.layers.33.mlp.down_proj
- model.layers.34.mlp.down_proj
- model.layers.35.mlp.down_proj
- model.layers.36.mlp.down_proj
- model.layers.37.mlp.down_proj
- model.layers.38.mlp.down_proj
- model.layers.39.mlp.down_proj
- model.layers.40.mlp.down_proj
# mlp.gate_proj layers
- model.layers.51.mlp.gate_proj
- model.layers.50.mlp.gate_proj
- model.layers.53.mlp.gate_proj
- model.layers.52.mlp.gate_proj
- model.layers.49.mlp.gate_proj
- model.layers.45.mlp.gate_proj
- model.layers.46.mlp.gate_proj
- model.layers.47.mlp.gate_proj
- model.layers.57.mlp.gate_proj
- model.layers.48.mlp.gate_proj
- model.layers.56.mlp.gate_proj
- model.layers.41.mlp.gate_proj
- model.layers.54.mlp.gate_proj
- model.layers.43.mlp.gate_proj
- model.layers.44.mlp.gate_proj
- model.layers.60.mlp.gate_proj
- model.layers.55.mlp.gate_proj
- model.layers.40.mlp.gate_proj
- model.layers.42.mlp.gate_proj
- model.layers.58.mlp.gate_proj
- model.layers.36.mlp.gate_proj
- model.layers.37.mlp.gate_proj
- model.layers.38.mlp.gate_proj
- model.layers.39.mlp.gate_proj
# mlp.up_proj layers
- model.layers.50.mlp.up_proj
- model.layers.51.mlp.up_proj
- model.layers.41.mlp.up_proj
- model.layers.49.mlp.up_proj
- model.layers.43.mlp.up_proj
- model.layers.44.mlp.up_proj
- model.layers.40.mlp.up_proj
- model.layers.45.mlp.up_proj
- model.layers.47.mlp.up_proj
- model.layers.48.mlp.up_proj
- model.layers.46.mlp.up_proj
- model.layers.42.mlp.up_proj
- model.layers.39.mlp.up_proj
- model.layers.36.mlp.up_proj
- model.layers.37.mlp.up_proj
- model.layers.38.mlp.up_proj
- model.layers.56.mlp.up_proj
- model.layers.57.mlp.up_proj
- model.layers.53.mlp.up_proj
- model.layers.31.mlp.up_proj
- model.layers.32.mlp.up_proj
- model.layers.34.mlp.up_proj
- model.layers.35.mlp.up_proj
- model.layers.33.mlp.up_proj
# model.embed_tokens layers
# model.norm layers
# post_attention_layernorm layers
- model.layers.0.post_attention_layernorm
- model.layers.1.post_attention_layernorm
- model.layers.2.post_attention_layernorm
- model.layers.3.post_attention_layernorm
- model.layers.4.post_attention_layernorm
- model.layers.5.post_attention_layernorm
- model.layers.6.post_attention_layernorm
- model.layers.7.post_attention_layernorm
- model.layers.8.post_attention_layernorm
- model.layers.9.post_attention_layernorm
- model.layers.10.post_attention_layernorm
- model.layers.11.post_attention_layernorm
- model.layers.12.post_attention_layernorm
- model.layers.13.post_attention_layernorm
- model.layers.14.post_attention_layernorm
- model.layers.15.post_attention_layernorm
- model.layers.16.post_attention_layernorm
- model.layers.17.post_attention_layernorm
- model.layers.18.post_attention_layernorm
- model.layers.19.post_attention_layernorm
- model.layers.20.post_attention_layernorm
- model.layers.21.post_attention_layernorm
- model.layers.22.post_attention_layernorm
- model.layers.23.post_attention_layernorm
# self_attn.k_proj layers
- model.layers.42.self_attn.k_proj
- model.layers.41.self_attn.k_proj
- model.layers.39.self_attn.k_proj
- model.layers.35.self_attn.k_proj
- model.layers.28.self_attn.k_proj
- model.layers.79.self_attn.k_proj
- model.layers.43.self_attn.k_proj
- model.layers.32.self_attn.k_proj
- model.layers.73.self_attn.k_proj
- model.layers.31.self_attn.k_proj
- model.layers.29.self_attn.k_proj
- model.layers.76.self_attn.k_proj
- model.layers.30.self_attn.k_proj
- model.layers.40.self_attn.k_proj
- model.layers.33.self_attn.k_proj
- model.layers.78.self_attn.k_proj
- model.layers.34.self_attn.k_proj
- model.layers.37.self_attn.k_proj
- model.layers.45.self_attn.k_proj
- model.layers.44.self_attn.k_proj
- model.layers.71.self_attn.k_proj
- model.layers.26.self_attn.k_proj
- model.layers.74.self_attn.k_proj
- model.layers.27.self_attn.k_proj
# self_attn.o_proj layers
- model.layers.35.self_attn.o_proj
- model.layers.34.self_attn.o_proj
- model.layers.37.self_attn.o_proj
- model.layers.33.self_attn.o_proj
- model.layers.31.self_attn.o_proj
- model.layers.27.self_attn.o_proj
- model.layers.38.self_attn.o_proj
- model.layers.24.self_attn.o_proj
- model.layers.39.self_attn.o_proj
- model.layers.43.self_attn.o_proj
- model.layers.29.self_attn.o_proj
- model.layers.0.self_attn.o_proj
- model.layers.50.self_attn.o_proj
- model.layers.32.self_attn.o_proj
- model.layers.45.self_attn.o_proj
- model.layers.30.self_attn.o_proj
- model.layers.60.self_attn.o_proj
- model.layers.23.self_attn.o_proj
- model.layers.18.self_attn.o_proj
- model.layers.67.self_attn.o_proj
- model.layers.57.self_attn.o_proj
- model.layers.20.self_attn.o_proj
- model.layers.76.self_attn.o_proj
- model.layers.28.self_attn.o_proj
# self_attn.q_proj layers
- model.layers.1.self_attn.q_proj
- model.layers.6.self_attn.q_proj
- model.layers.0.self_attn.q_proj
- model.layers.5.self_attn.q_proj
- model.layers.2.self_attn.q_proj
- model.layers.7.self_attn.q_proj
- model.layers.3.self_attn.q_proj
- model.layers.4.self_attn.q_proj
- model.layers.8.self_attn.q_proj
- model.layers.9.self_attn.q_proj
- model.layers.61.self_attn.q_proj
- model.layers.10.self_attn.q_proj
- model.layers.62.self_attn.q_proj
- model.layers.36.self_attn.q_proj
- model.layers.15.self_attn.q_proj
- model.layers.11.self_attn.q_proj
- model.layers.17.self_attn.q_proj
- model.layers.60.self_attn.q_proj
- model.layers.63.self_attn.q_proj
- model.layers.64.self_attn.q_proj
- model.layers.29.self_attn.q_proj
- model.layers.30.self_attn.q_proj
- model.layers.55.self_attn.q_proj
- model.layers.34.self_attn.q_proj
# self_attn.v_proj layers
- model.layers.12.self_attn.v_proj
- model.layers.16.self_attn.v_proj
- model.layers.18.self_attn.v_proj
- model.layers.19.self_attn.v_proj
- model.layers.20.self_attn.v_proj
- model.layers.21.self_attn.v_proj
- model.layers.22.self_attn.v_proj
- model.layers.23.self_attn.v_proj
- model.layers.24.self_attn.v_proj
- model.layers.25.self_attn.v_proj
- model.layers.26.self_attn.v_proj
- model.layers.27.self_attn.v_proj
- model.layers.28.self_attn.v_proj
- model.layers.29.self_attn.v_proj
- model.layers.30.self_attn.v_proj
- model.layers.31.self_attn.v_proj
- model.layers.32.self_attn.v_proj
- model.layers.33.self_attn.v_proj
- model.layers.34.self_attn.v_proj
- model.layers.35.self_attn.v_proj
- model.layers.36.self_attn.v_proj
- model.layers.37.self_attn.v_proj
- model.layers.38.self_attn.v_proj
- model.layers.39.self_attn.v_proj



sequence_len: 8192 # supports up to 8192
sample_packing: true
pad_to_sequence_len: true

# adapter: lora
# lora_model_dir:
# lora_r: 32
# lora_alpha: 16
# lora_dropout: 0.05
# lora_target_linear: true
# lora_fan_in_fan_out:

wandb_project: dolphin-2.9-qwen-1.5-110b
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 1
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 1e-5

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: true

gradient_checkpointing: true
early_stopping_patience:
# resume_from_checkpoint: /workspace/axolotl/qwen-checkpoint
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
eval_max_new_tokens: 128
saves_per_epoch: 4
save_total_limit: 2
debug:
deepspeed: deepspeed_configs/zero3_bf16_cpuoffload_params.json
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  eos_token: "<|im_end|>"

    ''',
    'model_training_hyperparameters': '''
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 1
    ''',
    'model_training_result': '''
| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.3528        | 0.0   | 1    | 0.3848          |
| 0.3687        | 0.25  | 291  | 0.3988          |
| 0.4156        | 0.5   | 582  | 0.3966          |
| 0.3826        | 0.75  | 873  | 0.3931          |
    ''',
    'model_training_framework_version': '''
- Transformers 4.40.0.dev0
- Pytorch 2.2.2+cu121
- Datasets 2.15.0
- Tokenizers 0.15.0
    '''
}

# Render the template with the model details
model_card = template.render(model_details)

# Print the generated model card
print(model_card)