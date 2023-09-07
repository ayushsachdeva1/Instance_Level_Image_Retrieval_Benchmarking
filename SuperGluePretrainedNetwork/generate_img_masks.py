import pickle as pkl
import cv2

input_dir = "/home/as216/segment_anything_amur/train_masks.pkl"
output_dir = '/scratch/as216/amur/image_masks/test/mask_'

def main():

    with open(input_dir, 'rb') as handle:
        masks = pkl.load(handle)

    for elem in masks:
        print(elem[0])
        mask = elem[1]*255
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)
        cv2.imwrite(output_dir + elem[0] + '.jpg', mask_image)

if __name__ == '__main__':
    main()
