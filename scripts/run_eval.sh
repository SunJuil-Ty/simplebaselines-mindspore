export DEVICE_ID=$1

python eval.py > eval_log$1.txt 2>&1 &
