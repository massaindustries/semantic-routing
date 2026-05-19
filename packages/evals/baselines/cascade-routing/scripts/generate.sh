python scripts/classification_inference.py --dataset arc
python scripts/classification_inference.py --dataset mmlu
python scripts/classification_inference.py --dataset mixeval
python scripts/free_form_inference.py --dataset mmlu
python scripts/free_form_inference.py --dataset gsm8k

cd data
wget https://huggingface.co/datasets/withmartian/routerbench/blob/main/routerbench_0shot.pkl
wget https://huggingface.co/datasets/withmartian/routerbench/blob/main/routerbench_5shot.pkl
cd ..

python scripts/preprocess.py