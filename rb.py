import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from ipywidgets import IntProgress

from cloths_segmentation.pre_trained_models import create_model
import os

def create_list():
    path = "static/data/"
    dir_list = os.listdir(path)
    return dir_list


def run_seg():
  model = create_model("Unet_2020-10-30")
  model.eval()

  for i in create_list():
    new_name=i
    new_name=new_name.replace("jpg","png")
    new_name=new_name.replace("jpeg","png")
    path="static/data/"+i
    image = cv2.imread(path)
    image_2_extract = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
      prediction = model(x)[0][0]
      mask = (prediction > 0).cpu().numpy().astype(np.uint8)
      mask = unpad(mask, pads)
      rmask = (cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) * 255).astype(np.uint8)
      mask2 = np.where((rmask < 255), 0, 1).astype('uint8')
      image_2_extract = image_2_extract * mask2[:, :, 1, np.newaxis]

      tmp = cv2.cvtColor(image_2_extract, cv2.COLOR_BGR2GRAY)
      _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
      b, g, r = cv2.split(image_2_extract)
      rgba = [b, g, r, alpha]
      dst = cv2.merge(rgba, 4)
      output_name="static/segmentation/"+str(new_name)
      cv2.imwrite(output_name, dst)

