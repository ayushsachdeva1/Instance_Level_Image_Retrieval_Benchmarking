import torch
import torchvision
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pickle

from segment_anything import SamPredictor, sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

owl_vit_file = '/home/as216/scenic/train_amur_owl_vit_results.pkl'
image_dir = '/scratch/as216/amur/train/0'

with open(owl_vit_file, 'rb') as f:
     train = pickle.load(f)  

# Download checkpoint from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

results = []
count = 0

for elem in train:
    img_file = image_dir + elem['img'] + '.jpg'
    logits, scores, labels, boxes = elem['results']

    if os.path.isfile(img_file) == False:
        continue
    
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    count += 1

    predictor.set_image(image)

    ind3, ind2, ind1 = np.argsort(scores)[-3:]

    w, h, _ = image.shape
    cx, cy, w, h = boxes[ind1]*max(w, h)

    input_box = np.array([cx - w / 2, cy - h/2, cx + w/2, cy + h/2])


    masks, iou_predictions, low_res_logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    results.append(("0" + elem['img'], masks[0], iou_predictions[0]))

    if count < 10:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.savefig('img_' + str(iou_predictions[0]) + "_" + elem['img'] + '.png')


with open('train_masks.pkl', 'wb') as handle:
    pickle.dump(results, handle)

print(count)


