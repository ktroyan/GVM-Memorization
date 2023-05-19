# Readme


## Generated Images

### Historic Data

- SD ver 2.1: [[images in .zip](https://drive.google.com/file/d/1d4zHZ0sBdjSomjjS137DTP-yhEAWJ4Vh/view?usp=sharing)]  [[images stored in .npy](https://drive.google.com/file/d/1hVLcaKpu-CuVEZB7Kq9QGfUfCqrhT8yt/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1_Y0WiH7Pac_5OOyzkALoaG8NonAedipf/view?usp=sharing)]
- SD ver 1.5: [[images in .zip](https://drive.google.com/file/d/1ceQyy9kAdUHq1HGbcaW1fsLFi9yQPTWz/view?usp=share_link)] [[image descriptions](https://drive.google.com/file/d/1VXuwaNIYVt1JFUIh6A5p65AIHFM0HHzJ/view?usp=sharing)]

- CLIP Image embeddings:
    - SD ver 2.1, Torch Tensors: [[openai_ViT-B-32](https://drive.google.com/file/d/1AAqHDVlV2RxVa7jGCT_iOPn3zvaZ7OOE/view?usp=share_link)] [[laion2b_s34b_b79k_ViT-B-32](https://drive.google.com/file/d/1cH5PHM725ILvxTKaiLnlVv7UFMOtlt2U/view?usp=share_link)]
    - SD ver 1.5, Torch Tensors: [[openai_ViT-B-32](https://drive.google.com/file/d/1ERCr626pc5xWgqrEKHOX3Pk1TG2K7kOg/view?usp=share_link)] [[laion2b_s34b_b79k_ViT-B-32](https://drive.google.com/file/d/1KPF96KTau66zLPWw3xYPtxrYwD-fey1e/view?usp=sharing)]     

### ArtStation Data:
- SD ver 1.5: [[images in .zip](https://drive.google.com/file/d/1jn7brUSN1peBnqo3JG_LT5sBg6rPHNPm/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1Cfz2JjR8V4U5Ct_mi1uqphRzSem9gtj5/view?usp=sharing)]

- CLIP Image embeddings:
    - SD ver 1.5, Torch Tensors: [[openai_ViT-B-32](https://drive.google.com/file/d/1FCGXLnFb_lyLvqx37E8R9n5TIB7o7Wcu/view?usp=sharing)] [[laion2b_s34b_b79k_ViT-B-32](https://drive.google.com/file/d/1L8rF1pv_FjGvkvFXoECat2Qhug29XpG3/view?usp=sharing)]     

> Load embeddings with `torch.load('sd_2_1_openai_ViT-B-32.pt')`

## Zero-shot learning on historic images

Number of real images: 1181    
Number of generated images: 3735 (249 artists x 15 images)

### On generated images
|SD version|Image embedding|Top-1|Top-5|  
|---|---|---|---|
|2.1|openai + ViT-B-32|5.78|17.54|
|2.1|laion2b_s34b_b79k + ViT-B-32|5.49|15.1|
|2.1|laion400m_e32 + ViT-B-32 |5.11|15.21|
|1.5|openai + ViT-B-32|10.15|25.46|
|1.5|laion2b_s34b_b79k + ViT-B-32|10.23|26.02|
|1.5|laion400m_e32 + ViT-B-32 |8.81|22.78|



### On real images
|Image embedding|Top-1|Top-5|  
|---|---|---|
|openai + ViT-B-32|52.75|77.05|
|laion2b_s34b_b79k + ViT-B-32|55.63|79.51|
|laion400m_e32 + ViT-B-32|45.89|67.49|

## Zero-shot learning on ArtStation images

Number of real images:     
Number of generated images: 3960 (264 artists x 15 images)

### On generated images
|SD version|Image embedding|Top-1|Top-5|  
|---|---|---|---|
|2.1|openai + ViT-B-32|0.88|3.74|
|2.1|laion2b_s34b_b79k + ViT-B-32|0.96|3.81|
|2.1|laion400m_e32 + ViT-B-32 |0.81|3.36|
|1.5|openai + ViT-B-32|1.31|4.24|
|1.5|laion2b_s34b_b79k + ViT-B-32|0.86|3.33|
|1.5|laion400m_e32 + ViT-B-32 |0.73|3.21|
