# reload utils
import sys
import os
# Add the parent directory to Python path more reliably
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from util import utils
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from util.box_annotator import BoxAnnotator



def draw_ocr_boxes_and_save(image_path, ocr_bbox, ocr_text, output_path='./annotated_image.png', text_scale=0.4, text_padding=5, text_thickness=2, thickness=3):
    """
    绘制OCR检测到的bounding box并保存图片，使用与get_som_labeled_img相同的逻辑
    
    Args:
        image_path: 输入图片路径
        ocr_bbox: OCR检测到的bounding box坐标列表，格式为[(x1,y1,x2,y2), ...]
        ocr_text: 对应的文本内容列表（仅用于过滤，不显示）
        output_path: 输出图片路径
        text_scale: 文本缩放比例
        text_padding: 文本内边距
        text_thickness: 文本粗细
        thickness: 边框粗细
    """
    # 读取图片
    image_source = Image.open(image_path).convert("RGB")
    w, h = image_source.size
    image_source = np.asarray(image_source)
    
    if not ocr_bbox:
        print("No OCR bounding boxes found!")
        return
    
    # 转换为tensor并归一化到0-1范围
    ocr_bbox_tensor = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
    
    # 转换为cxcywh格式（与get_som_labeled_img中的逻辑一致）
    boxes = box_convert(boxes=ocr_bbox_tensor, in_fmt="xyxy", out_fmt="cxcywh")
    
    # 创建虚拟的logits和phrases（与get_som_labeled_img中的逻辑一致）
    logits = torch.ones(len(boxes))  # 虚拟置信度
    phrases = [str(i) for i in range(len(boxes))]
    
    # 使用与get_som_labeled_img相同的annotate逻辑
    h, w, _ = image_source.shape
    boxes_pixel = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xywh").numpy()
    
    # 导入supervision用于检测
    import supervision as sv
    detections = sv.Detections(xyxy=xyxy)
    
    # 只显示数字ID，不显示文本内容（与get_som_labeled_img中的逻辑一致）
    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]
    
    # 使用BoxAnnotator绘制
    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding, text_thickness=text_thickness, thickness=thickness)
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h))
    
    # 保存图片
    pil_img = Image.fromarray(annotated_frame)
    pil_img.save(output_path)
    print(f"Annotated image saved to: {output_path}")
    
    # 返回坐标信息（与get_som_labeled_img中的逻辑一致）
    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates



# importlib.reload(utils)
# from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

image_path = 'imgs/google_page.png'
image_path = 'imgs/windows_home.png'
# image_path = 'imgs/windows_multitab.png'
# image_path = 'imgs/omni3.jpg'
# image_path = 'imgs/ios.png'
image_path = 'imgs/word.png'
# image_path = 'imgs/excel2.png'
image_path = 'lzy/test.png'

image = Image.open(image_path)
image_rgb = image.convert('RGB')
print('image size:', image.size)



box_overlay_ratio = max(image.size) / 3200
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}
BOX_TRESHOLD = 0.05


easyocr_args={'paragraph': True, 'text_threshold':0.7, 'width_ths':20.0}


ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
    image_path, 
    display_img = False, 
    output_bb_format='xyxy', 
    goal_filtering=None, 
    easyocr_args=easyocr_args, 
    use_paddleocr=False,
    )
text, ocr_bbox = ocr_bbox_rslt



# 将easyocr_args转换为列表格式并拼接到output_path
easyocr_list = [f"{k}_{v}" for k, v in easyocr_args.items()]
easyocr_suffix = "_".join(easyocr_list)
output_path = f'./annotated_image_{easyocr_suffix}.png'



# 调用函数绘制OCR bounding box
if ocr_bbox and text:
    draw_ocr_boxes_and_save(image_path, ocr_bbox, text, output_path)
    print("need batch processing")
else:
    print("No OCR bounding boxes found!")




