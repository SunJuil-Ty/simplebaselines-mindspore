"""
export simple_baseline to mindir or air
"""
import argparse
import numpy as np
from mindspore import context, Tensor, export
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.pose_resnet import GetPoseResNet

parser = argparse.ArgumentParser(description='simple_baselines')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_url", default="/home/dataset/coco/multi_train_poseresnet_commit_0-140_292.ckpt",
                    help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="simple_baselines", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["MINDIR", "ONNX"], default='MINDIR', help='file format')
args = parser.parse_args()

if __name__ == '__main__':
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
        save_graphs=False,
        device_id=args.device_id)

    pose_res_net = GetPoseResNet(config)
    pose_res_net.set_train(False)

    print('loading model ckpt from {}'.format(args.ckpt_url))
    load_checkpoint(args.ckpt_url)
    load_param_into_net(pose_res_net, load_checkpoint(args.ckpt_url))

    input_data = Tensor(np.zeros([1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]]), mstype.float32)
    export(pose_res_net, input_data, file_name=args.file_name, file_format=args.file_format)
    
