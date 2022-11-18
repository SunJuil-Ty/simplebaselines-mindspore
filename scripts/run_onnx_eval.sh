if [ $# -lt 2 ]
then
    echo "Usage: bash scripts/run_onnx_eval.sh [CKPT_FILES] [DEVICE_TYPE]"
    exit 1
fi

CKPT_FILES=$1
DEVICE_TYPE=$2

python eval_onnx.py --ckpt_path=$CKPT_FILES \
    --target_device=$DEVICE_TYPE > output.eval_onnx.log 2>&1 &
