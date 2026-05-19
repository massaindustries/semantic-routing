python scripts/swebench_run.py --models 0,1,2,3,9 --half-test
python scripts/swebench_run.py --models 0,1,2,3,4,5,6,7,8,9 --half-test

python scripts/code_math.py --models 0,1,2,3 --half-test

python scripts/classification.py --models 0,1,2 &
python scripts/classification.py --models 3,4,5 & 
python scripts/classification.py --models 6,7,8

python scripts/free_form.py --models 0,1,2 &
python scripts/free_form.py --models 3,4,5 &
python scripts/free_form.py --models 6,7,8


python scripts/routerbench.py --models 9,4,5 --noise-level low
python scripts/routerbench.py --models 9,4,5 --noise-level medium &
python scripts/routerbench.py --models 9,4,5 --noise-level high &
python scripts/routerbench.py --models 0,9,4,3,5 --noise-level low 
python scripts/routerbench.py --models 0,9,4,3,5 --noise-level medium &
python scripts/routerbench.py --models 0,9,4,3,5 --noise-level high &
python scripts/routerbench.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level low
python scripts/routerbench.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level medium &
python scripts/routerbench.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level high &
python scripts/routerbench.py --models 9,4,5 --noise-level low --few-shot
python scripts/routerbench.py --models 9,4,5 --noise-level medium --few-shot &
python scripts/routerbench.py --models 9,4,5 --noise-level high --few-shot &
python scripts/routerbench.py --models 0,9,4,3,5 --noise-level low  --few-shot
python scripts/routerbench.py --models 0,9,4,3,5 --noise-level medium --few-shot &
python scripts/routerbench.py --models 0,9,4,3,5 --noise-level high --few-shot &
python scripts/routerbench.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level low  --few-shot
python scripts/routerbench.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level medium --few-shot &
python scripts/routerbench.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level high --few-shot &

python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level medium
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level medium --no-speedup &
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level medium --greedy &
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level medium --sigma-none

python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level low &
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level low --no-speedup &
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level low --greedy
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level low --sigma-none &

python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level high &
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level high --no-speedup
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level high --greedy &
python scripts/routerbench_times.py --models 0,1,2,3,4,5,6,7,8,9,10 --noise-level high --sigma-none &

bash scripts/routerbench_ablation.sh

python scripts/runtime.py