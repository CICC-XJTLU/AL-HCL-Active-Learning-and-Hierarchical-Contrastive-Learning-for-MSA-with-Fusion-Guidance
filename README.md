# AL-ARCF

Pytorch implementation for codes in "AL-HCL: Active Learning and Hierarchical Contrastive Learning for Multimodal Sentiment Analysis with Fusion Guidance"(https://ieeexplore.ieee.org/document/11180049)

![image-20250930140223812](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20250930140223812.png)

# Prepare

## Datasets

Download the pkl file (https://drive.google.com/drive/folders/1_u1Vt0_4g0RLoQbdslBwAdMslEdW1avI?usp=sharing). Put it under the "./dataset" directory.

## Pre-trained language model

Download the SentiLARE language model files (https://drive.google.com/file/d/1onz0ds0CchBRFcSc_AkTLH_AZX_iNTjO/view?usp=share_link), and then put them into the "./pretrained-model/sentilare_model" directory.

# Run

''' python train.py '''

# Paper

Please cite our paper if you find our work useful for your research:

```
@ARTICLE{11180049,
  author={He, Xiaojiang and Pan, Yushan and Xu, Zhijie and Li, Zuhe and Guo, Xinfei and Yang, Chenguang},
  journal={IEEE Transactions on Affective Computing}, 
  title={AL-HCL: Active Learning and Hierarchical Contrastive Learning for Multimodal Sentiment Analysis with Fusion Guidance}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Contrastive learning;Active learning;Sentiment analysis;Feature extraction;Visualization;Data models;Vectors;Uncertainty;Affective computing;Data mining;Multimodal Sentiment Analysis;Active Learning;Contrastive Learning},
  doi={10.1109/TAFFC.2025.3614159}}


```

