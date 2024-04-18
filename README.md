## CAAM (Cross Attention is All for Multimodal ; CAAM)
<br/>
<p align="center"> <img src="./source/architecture.png" width="700" height="400">

The video's image and audio data, as well as caption text data describing the video, are passed through an encoder that can represent each data type well to generate a feature value.
<br/> And we map these feature values to the joint embedding space via a projection head.
<br/> After that, a final embedding vector is generated by considering the **<span style="color:#D0E4FC"> correlation between modalities</span>** through **<span style="color:#D0E4FC">a multi-modal fusion transformer</span>**. 
<br/>Finally, we put this embedding vector into a classifier and proceeded with a <span style="color:#D0E4FC"> **classification task** </span> to determine which of the 20 categories the video belongs to.


# Dataset 
### MSR VTT ###
- Original dataset: [download](https://www.dropbox.com/sh/bd75sz4m734xs0z/AADbN9Ujhn6FZX12ulpNWyR_a?dl=0)
- Our dataset: [download](https://drive.google.com/drive/folders/1JsGZKp3ZAoC7w2XaOkZp4TnQ0GwGwUtU?usp=sharing)

MSR-VTT is a large dataset constructed for video captioning tasks, consisting of 10,000 video clips corresponding to 20 categories.
Each video clip is annotated with 20 English sentences.
The dataset used 6513 training sets (train-set) and 497 validation sets (Valuation-set).
**We used the extracted feature value for learning**

# Training 
This has trained the model with default settings, RTX 4080 16GB GPU memory occupied, batch size 16, 200 epoch 
<br/>
 ```bash
python train.py \
--we_path 'data/GoogleNews-vectors-negative300.bin' \
--data_path 'data/msrvtt_category_train.pkl'\
--val_data_path 'data/msrvtt_category_test.pkl' \
--save_path 'weights_classifier' \
--exp trial1 \
--epoch 200 \
--use_softmax True \
--use_cls_token False \
--token_projection 'projection_net' \
--num_classes 20 \
--batch_size 16 \
--device "0"

```
## Result
| # Cross Attention | Accuracy |                                                                   
|-------------------|------------|
| 3                 | 55.95          |    
| 6 (Fix K)         | 54.66          |   
| **6**                 | **59.11*    |    

- 3 cross-attention experiments
- 6 cross-attention experiments (experiments with three modalities / keys, values fixed)
- 6 cross-attention (an experiment in which key values were alternately used in pairs of 2 modalities)

## Evaluation

| # Method                                 | # Modality                                        | Acc @1%                                       | Acc @5%                                    |
|------------------------------------------|---------------------------------------------------|-----------------------------------------------|--------------------------------------------|
| ViT(video)                               | V                                                 | 53.7                                          | 82.9                                       |
| GRU                                      | V                                                 | 49.5                                          | 79.2                                       |
| MCA-WF(GRU)                              | V                                                 | 53.8                                          | 83.8                                       |
| GRU                                      | V + T                                             | 53.1                                          | 81.8                                       |
| MCA-WF(GRU)                              | V + T                                             | 56.4                                          | 84.02                                      |
| ViLT                                     | V + T                                             | 55.4                                          | 83.9                                       |
| MCA-WF(ViLT)                             | V + T                                             | 58.8                                          | 85.3                                       |
| Ours-CA3                                 | V + T + A                                         | 55.95                                         | -                                          |
| Ours-Fix K                               | V + T + A                                         | 54.66                                         | -                                          |
| <span style="color:#D0E4FC">**Ours** </span> | <span style="color:#D0E4FC">**V + T + A** </span> | <span style="color:#D0E4FC">**59.11** </span> | <span style="color:#D0E4FC"> **-** </span> |
