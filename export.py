import os
import argparse
import torch
import tensorrt as trt
import onnx
from pathlib import Path
import sys
from utils.net import ResNet18 as Model  # 导入你的模型定义
from utils.general import select_device  # 复用设备选择工具

# TensorRT日志设置
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')  # 初始化TensorRT插件

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def export_onnx(model, im, output_path, opset=12, dynamic=False):
    """
    导出ONNX模型
    :param model: 加载权重的PyTorch模型
    :param im: 输入张量（用于确定输入形状）
    :param output_path: ONNX输出路径
    :param opset: ONNX算子集版本
    :param dynamic: 是否启用动态批次维度
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 模型切换到推理模式
    model.eval()
    
    # 动态维度设置（如果需要）
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},  # 输入批次维度动态
            'output': {0: 'batch_size'}   # 输出批次维度动态
        }
    
    # 导出ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            im,
            output_path,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['input'],  # 输入节点名称
            output_names=['output'],  # 输出节点名称
            dynamic_axes=dynamic_axes
        )
    
    # 验证ONNX模型有效性
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)  # 检查模型结构是否正确
    print(f"ONNX模型导出成功: {output_path}")
    return output_path


def export_engine(onnx_path, engine_path, precision='fp32'):
    """
    将ONNX模型转换为TensorRT Engine
    :param onnx_path: ONNX模型路径
    :param engine_path: TensorRT Engine输出路径
    :param precision: 精度模式 ('fp32'/'fp16')
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    
    # 创建TensorRT构建器和网络
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # 配置构建器
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB工作空间
        
        # 设置精度
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("启用FP16精度加速")
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("启用INT8精度加速（需校准，此处简化处理）")
        
        # 解析ONNX模型
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("ONNX解析失败:")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # 构建引擎
        print("开始构建TensorRT Engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            print("Engine构建失败")
            return None
        
        # 保存Engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"TensorRT Engine导出成功: {engine_path}")
        return engine_path


def main(opt):
    # 设备选择（TensorRT需要CUDA设备）
    device = select_device(opt.device)
    if device.type != 'cuda':
        raise ValueError("TensorRT导出需要CUDA设备，请指定--device cuda")
    
    # 加载模型
    model = Model().to(device)
    if not os.path.exists(opt.weights):
        raise FileNotFoundError(f"权重文件不存在: {opt.weights}")
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    print(f"已加载权重: {opt.weights}")
    
    # 创建示例输入（需与训练时输入形状一致，这里假设3通道224x224图像）
    im = torch.zeros(opt.batch_size, opt.img_size, opt.img_size, 3).to(device).float()
    print(f"使用输入形状: {im.shape}")
    
    # 导出ONNX
    onnx_path = os.path.join(opt.output, f"model_{opt.img_size}.onnx")
    export_onnx(
        model,
        im,
        onnx_path,
        opset=opt.opset,
        dynamic=opt.dynamic
    )
    
    # 导出TensorRT Engine
    if opt.engine:
        engine_path = os.path.join(opt.output, f"model_{opt.img_size}_{opt.precision}.engine")
        export_engine(
            onnx_path,
            engine_path,
            precision=opt.precision
        )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'E:\DMH\Shanghai\dmh_categorization\train\runs\exp2\weights\best.pt', help='训练好的权重路径（best.pt或last.pt）')
    parser.add_argument('--output', type=str, default=ROOT / 'export', help='导出文件保存目录')
    parser.add_argument('--img-size', type=int, default=224, help='输入图像尺寸（宽=高）')
    parser.add_argument('--batch-size', type=int, default=1, help='导出时的批次大小（动态维度时可设为1）')
    parser.add_argument('--dynamic', action='store_true', help='启用动态批次维度')
    parser.add_argument('--opset', type=int, default=12, help='ONNX算子集版本')
    parser.add_argument('--engine', action='store_true', default=True, help='是否导出TensorRT Engine')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'], help='TensorRT精度模式')
    parser.add_argument('--device', default='0', help='设备选择（必须为CUDA设备，如0或cuda）')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)