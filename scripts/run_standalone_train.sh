echo "========================================================================"
echo "Please run the script as: "
echo "bash run_standalone_train.sh"
echo "For example: bash run_standalone_train.sh"
echo "It is better to use the absolute path."
echo "========================================================================"
echo "start training for device $DEVICE_ID"
export DEVICE_ID=$1
python -u ../train.py --device_id ${DEVICE_ID} --is_model_arts False --run_distribute False > train${DEVICE_ID}.log 2>&1 &
echo "finish"
