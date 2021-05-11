from pathlib import Path
from second.pytorch.train import train, evaluate
from second.protos import pipeline_pb2
from google.protobuf import text_format
from second.utils import config_tool


def train_multi_rpn_layer_num():
    config_path = "./configs/car.lite.config" # Configuration path
    model_root = Path.home() / "second_test" # Root of the model to perform second test
    config = pipeline_pb2.TrainEvalPipelineConfig() 
    
    # Open the config file 
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second

    layer_nums = [2, 4, 7, 9]

    for l in layer_nums:
        model_dir = str(model_root / f"car_lite_L{l}")
        model_cfg.rpn.layer_nums[:] = [l]
        train(config, model_dir)


def eval_multi_threshold():
    config_path = "./configs/car.fhd.config" # Configuration path
    ckpt_name = "/path/to/your/model_ckpt" # Path to model checkpoint
    
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    thresh = 0.3

    model_cfg.nms_score_threshold = thresh
    result_path = Path.home() / f"second_test_eval_{thresh:.2f}"
    evaluate(
        config,
        result_path=result_path,
        ckpt_path=str(ckpt_name),
        batch_size=1,
        measure_time=True)

if __name__ == "__main__":
    eval_multi_threshold()