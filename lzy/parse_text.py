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
from argparse import ArgumentParser
import json
import easyocr
import time
from tqdm import tqdm
from glob import glob
import random   


def draw_ocr_boxes_and_save(image_path, ocr_bbox, ocr_text, output_path='./annotated_image.png', text_scale=0.4, text_padding=5, text_thickness=1, thickness=1):
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


def create_dict_mapping(image_paths, mode, save_dir):
    dict_mapping_original_to_output = {}
    dict_mapping_output_to_original = {}
    for image_path in image_paths:
        # Convert image_path to absolute path
        abs_image_path = os.path.abspath(image_path)
        output_img_path, output_text_path = get_output_path(abs_image_path, mode, save_dir)
        dict_mapping_original_to_output[abs_image_path] = (output_img_path, output_text_path)
        dict_mapping_output_to_original[output_img_path] = abs_image_path
    total_dict = {
        "original_to_output": dict_mapping_original_to_output,
        "output_to_original": dict_mapping_output_to_original
    }

    # Convert save_dir to absolute path
    abs_save_dir = os.path.abspath(save_dir)
    with open(os.path.join(abs_save_dir, f'parsed_mode-{mode}-mapping_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(total_dict, f, indent=4, ensure_ascii=False)



def get_output_path(image_path, mode, save_dir):
    # Convert save_dir to absolute path
    abs_save_dir = os.path.abspath(save_dir)
    
    # 创建输出目录
    output_dir_img = os.path.join(abs_save_dir, 'parsed_images', f'parsed_mode-{mode}')
    output_dir_text = os.path.join(abs_save_dir, 'parsed_text_coordinates', f'parsed_mode-{mode}')
    
    # 获取输入图片的文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    output_img_path = os.path.abspath(os.path.join(output_dir_img, f'name-{input_filename}.png'))
    assert os.path.exists(output_img_path)
    
    output_text_path = os.path.abspath(os.path.join(output_dir_text, f'name-{input_filename}.json'))
    assert os.path.exists(output_text_path)
    
    return output_img_path, output_text_path

    




def parse_image_func(image_path, mode, save_dir):

    # 创建输出目录
    output_dir_img = os.path.join(save_dir, 'parsed_images', f'parsed_mode-{mode}')
    os.makedirs(output_dir_img, exist_ok=True)
    output_dir_text = os.path.join(save_dir, 'parsed_text_coordinates', f'parsed_mode-{mode}')
    os.makedirs(output_dir_text, exist_ok=True)
    
    # 获取输入图片的文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 构建输出路径：项目根目录/parsed_text_images/原文件名_mode.png
    output_path = os.path.join(output_dir_img, f'name-{input_filename}.png')
    if os.path.exists(output_path):
        return


    mode_to_args = {
        "paragraph": {'paragraph': True, 'text_threshold':0.7, 'width_ths':20.0},
        "word": {'paragraph': False, 'text_threshold':0.7, 'width_ths':0.1},
        "line": {'paragraph': False, 'text_threshold':0.7, 'width_ths':1.0},
    }

    easyocr_args = mode_to_args[mode]

    easyocr_args['batch_size'] = 50

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path, 
        display_img = False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args=easyocr_args, 
        use_paddleocr=False,
        )
    text, ocr_bbox = ocr_bbox_rslt

    # Save the image with ocr_bbox
    # 调用函数绘制OCR bounding box
    if ocr_bbox and text:
        draw_ocr_boxes_and_save(image_path, ocr_bbox, text, output_path)
        print("need batch processing")
    else:
        print("No OCR bounding boxes found!")
        raise ValueError("No OCR bounding boxes found!")


    # save the parsed text
    dict_text_coordinate = {}
    for i, text in enumerate(text):
        dict_text_coordinate[i] = {
            'text': text,
            'coordinate': ocr_bbox[i]
        }
    with open(os.path.join(output_dir_text, f'name-{input_filename}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_text_coordinate, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    

    parser = ArgumentParser()
    parser.add_argument("--image_path",'-i', type=str, default='imgs/google_page.png')
    parser.add_argument("--image_dir",'-id', type=str, default='imgs/')
    parser.add_argument("--single_image", action='store_true', default=False)
    parser.add_argument("--mode",'-m', type=str, default='paragraph', choices=['paragraph', 'word', 'line'])
    parser.add_argument("--save_dir",'-s', type=str, default='./parsed_results')
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--split_num", type=int, default=4)
    parser.add_argument("--create_mapping_dict", action='store_true', default=False)
    args = parser.parse_args()



    if args.single_image:
        parse_image_func(args.image_path, args.mode, args.save_dir)

    else:
        image_paths = glob(os.path.join(args.image_dir, '**/*.png'))
        if args.debug:
            debug_num = 100
            divider = len(image_paths) // debug_num
            image_paths = image_paths[::divider]

        if args.create_mapping_dict:
            create_dict_mapping(image_paths, args.mode, args.save_dir)
            exit()

        if args.split_index == -1:
            for image_path in tqdm(image_paths, desc = 'OCR processing'):
                parse_image_func(image_path, args.mode, args.save_dir)

        else:
            assert args.split_index < args.split_num
            image_paths = [image_path for index, image_path in enumerate(image_paths) if index % args.split_num == args.split_index]

            for image_path in tqdm(image_paths, desc = 'OCR processing'):
                parse_image_func(image_path, args.mode, args.save_dir)