# Reranking Transformer

These are the major changes to be made to the [Reranking Transformer repo](https://github.com/uvavision/RerankingTransformer/tree/main). 

We assume that the amur dataset or synthetic amur dataset is at /scratch/as216/amur. Also, we assume that the local and global descriptors (as generated using the DELG model) are at data/amur_train or data/synthetic_amur respectively,

With these additions, the finetune.py file can be successfully run to finetune the transformer (specifically the reranking transformer, not the one used in generating keypoints/descriptors).

To evaluate the results of the finetuned model, save the model weights and use them for inference on the test dataset and pass the results into the evaluation server code available [here](https://github.com/cvwc2019/ATRWEvalScript).
