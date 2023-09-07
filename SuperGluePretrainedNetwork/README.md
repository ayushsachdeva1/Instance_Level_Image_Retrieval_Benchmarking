# SuperGlue

These are the major changes to be made to the [SuperGlue repo](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master#reproducing-outdoor-evaluation-final-table). 

We assume that the amur dataset or synthetic amur dataset is at /scratch/as216/amur. If using masked images, we use that the pkl file with mask coordinates is at segment_anything_amur/train_masks.pkl.

With these additions, the match_amur.py file can be successfully run to run inference on the amur dataset. Note that this model uses the SuperPoint detector for keypoints.
If using masked images, first run generate_img_masks.py.

For results, pass the inference results into the evaluation server code available [here](https://github.com/cvwc2019/ATRWEvalScript).
