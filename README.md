# Benchmarking of Instance Retrieval Models on Amur Tiger Dataset
##### Work done under Dr. Vicente Ordoñez, [Vislang Group](https://vislang.ai/)

## Introduction
Wildlife Re-ID is the task of searching for the instance of a particular individual animal in a large database (usually using a query image). This is a specific case / variation of the generalized instance-level image retrieval task. The accuracy for this specific case, however, is lower than the general case due to the similarity between different individuals and lack of extensive training data. 

We want to benchmark the accuracy of state-of-the-art instance-level image retrieval models on a wildlife re-ID dataset for a better understanding of whether these generalized computer vision models are successful in extending to use cases that differ from the common landmark identification use case most of these models are trained for. If successful, this analysis would allow for these generalized models to replace species-specific models which are currently in use for conservation efforts in wildlife reserves/parks all across the world. Moreover, this would also allow for an understanding of how these models can generalize to data-scarce domains which differ from their original training domains.

We evaluate three model architectures that represent the current state-of-the-art in instance-level image retrieval: 
- Deep Local and Global features (DELG)
- SuperGlue
- Reranking Transformer

We will be finetuneing these models and evaluating them on the [Amur Dataset](https://arxiv.org/pdf/1906.05586.pdf). This dataset contains over 3000 images of 92 individual tigers. This dataset has been chosen as it allows for easy comparison of results with the leaderboard of an ICCV Workshop on the [dataset](https://cvwc2019.github.io/index.html#body-home). 

## Additions/Changes to Models
For fair comparison, we have at least one variant of the models where we have not made any changes to the core architecture of the model, except what is required to finetune/run inference on the amur dataset. 

Specifically, due to lack of original training weights, we have not finetuned the DELF/DELG model which is used to extract key points for DELG and for Reranking Transformer. Similarly, we have not been able to finetune Superglue because of the lack of original training weights for its keypoint extractor, SuperPoint. However, we extensively fine-tune the Reranking Transformer model which reranks the top-100 or top-500 or all of the retrieved images.

We also wanted to evaluate the performance of these models after we limited the extracted keypoints to only be from the segmented body of the tiger. To do this, we use a zero-shot segmentation model combining [OwlVIT](https://arxiv.org/pdf/2205.06230.pdf) and the [segment anything model](https://github.com/facebookresearch/segment-anything). We also trained a segmentation model specific to the amur dataset but have not reported the results as the use of this model restricts the dataset annotations to be more informative. 

Due to the lack of training data being a concern, we are currently working on evaluating these models with synthetic data to augment our training set. We are trying to recreate the image warping model from [DocUNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DocUNet_Document_Image_CVPR_2018_paper.pdf) to generate synthetic data with images of tigers. 

## Code
Unfortunately, repositories with code for the models are massive leading to inconvenience while combining them on github. So this repository will only contain files from the different repositories which are changed for finetuning or to work the amur dataset. 

There are READMEs in each of the folders explaining how to make the necessary changes to finetune and run inference using these models.

Note: For all code, we assume that the amur dataset is in a scratch folder at /scratch/as216/amur/. 

## Results
After Benchmarking, we find that the SOTA models (finetuned or with the original weights) do not come sufficiently close to the top of the leaderboard for the competition for us to conclude that these models can replace the species-specific models that segment the limbs, torso, head, tail, etc. and use other domain knowledge to build the model architecture. The results are compiled in the following table: 
| Model                               | mmAP  | Single Cam |       |       | Cross Cam |       |       |
|-------------------------------------|-------|:----------:|:-----:|:-----:|:---------:|:-----:|:-----:|
|                                     |       | mAP        | top-1 | top-5 | mAP       | top-1 | top-5 |
| DELF (Average local feature)        | 0.413 | 0.523      | 0.791 | 0.937 | 0.302     | 0.709 | 0.846 |
| DELF (RANSAC)                       | 0.315 | 0.516      | 0.794 | 0.894 | 0.274     | 0.686 | 0.823 |
| DELG                                | 0.53  | 0.707      | 0.894 | 0.957 | 0.353     | 0.697 | 0.862 |
| Superglue                           | 0.51  | 0.599      | 0.777 | 0.903 | 0.421     | 0.691 | 0.874 |
| RRT - top100                        | 0.532 | 0.700      | 0.86  | 0.94  | 0.364     | 0.697 | 0.869 |
| RRT - all                           | 0.539 | 0.694      | 0.866 | 0.937 | 0.383     | 0.703 | 0.869 |
| RRT - top100(finetuned)             | 0.548 | 0.717      | 0.894 | 0.963 | 0.379     | 0.731 | 0.897 |
| RRT - all (finetuned)               | 0.582 | 0.718      | 0.866 | 0.949 | 0.446     | 0.749 | 0.897 |
| Superglue (limited keypoints)       | 0.442 | 0.505      | 0.751 | 0.874 | 0.378     | 0.702 | 0.886 |
| RRT (limited keypoints)             | 0.385 | 0.508      | 0.765 | 0.834 | 0.262     | 0.686 | 0.829 |
| RRT (finetuned + limited keypoints) | 0.458 | 0.563      | 0.803 | 0.903 | 0.352     | 0.714 | 0.857 |
