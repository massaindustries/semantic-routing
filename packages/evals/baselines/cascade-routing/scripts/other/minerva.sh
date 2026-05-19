lm_eval --model hf     --model_args pretrained=Qwen/Qwen2.5-Coder-7B-Instruct     --tasks minerva_math     --batch_size 4  --output_path data --log_samples
lm_eval --model hf     --model_args pretrained=Qwen/Qwen2.5-Coder-1.5B-Instruct     --tasks minerva_math     --batch_size 4  --output_path data --log_samples
lm_eval --model hf     --model_args pretrained=Qwen/Qwen2.5-Math-7B-Instruct     --tasks minerva_math     --batch_size 4   --output_path data --log_samples
lm_eval --model hf     --model_args pretrained=Qwen/Qwen2.5-Math-1.5B-Instruct     --tasks minerva_math     --batch_size 4   --output_path data --log_samples
