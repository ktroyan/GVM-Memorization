# Readme

## Real Images

### Historic Data

- [[images in .zip](https://drive.google.com/file/d/1ZAegLoQ20XXI1tWDgOZFVJdEraMW6Bqd/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1ZX1GpSbKcLjbdA48TNVdEIPQ-FgaldUr/view?usp=sharing)]
- CLIP Image embeddings:
    - [[openai](https://drive.google.com/file/d/1HWUpYLGZErlhaahLladfFj0YuebhjY9s/view?usp=sharing)] [[laion2b](https://drive.google.com/file/d/10ZnfiCjCw-ELPcWCiU0sZO0lwmKdyOtH/view?usp=sharing)]

### ArtStation Data

- [[images in .zip](https://drive.google.com/file/d/1UNJHZuaD_0kiSe4Y8wfo40jWrkdDfwhL/view?usp=sharing)] [[image descriptions (* filtered non-ascii artist names)](https://drive.google.com/file/d/1ARh2s9LU1vCbmmJPn1UwXHdCpL1z7Dho/view?usp=sharing)]
- CLIP Image embeddings:
    - [[openai](https://drive.google.com/file/d/1r9_0fMKmhymtpGiKJG4znLUdFv1l0Bkt/view?usp=sharing)] [[laion2b](https://drive.google.com/file/d/1l6vQkRAY4F6XJHaCHPZiE9vDGO233Ngh/view?usp=sharing)]


## Generated Images

### Historic Data

__Ver1__ and __Ver2__ follow similar format but with different prompts. Both of them have same sizes.

__Ver1__

- SD ver 2.1: [[images in .zip](https://drive.google.com/file/d/1d4zHZ0sBdjSomjjS137DTP-yhEAWJ4Vh/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1_Y0WiH7Pac_5OOyzkALoaG8NonAedipf/view?usp=sharing)]
- SD ver 1.5: [[images in .zip](https://drive.google.com/file/d/1ceQyy9kAdUHq1HGbcaW1fsLFi9yQPTWz/view?usp=share_link)] [[image descriptions](https://drive.google.com/file/d/1VXuwaNIYVt1JFUIh6A5p65AIHFM0HHzJ/view?usp=sharing)]

- CLIP Image embeddings:
    - SD ver 2.1, Torch Tensors: [[openai](https://drive.google.com/file/d/1AAqHDVlV2RxVa7jGCT_iOPn3zvaZ7OOE/view?usp=share_link)] [[laion2b](https://drive.google.com/file/d/1cH5PHM725ILvxTKaiLnlVv7UFMOtlt2U/view?usp=share_link)]
    - SD ver 1.5, Torch Tensors: [[openai](https://drive.google.com/file/d/1ERCr626pc5xWgqrEKHOX3Pk1TG2K7kOg/view?usp=share_link)] [[laion2b](https://drive.google.com/file/d/1KPF96KTau66zLPWw3xYPtxrYwD-fey1e/view?usp=sharing)]     

__Ver2__


- SD ver 2.1: [[images in .zip](https://drive.google.com/file/d/1Jgyu7Rxi6sQ2I7H0njeqNqPhT1NsPwKR/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1JNqughC6irSygR6HMKiCXplXORWEM_eL/view?usp=sharing)]
- SD ver 1.5: [[images in .zip](https://drive.google.com/file/d/11ttJP96Z6XePuR23qaV6bF6-0C_y6cA9/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1Gl7vD6u1487AjKoOJVGj7jWOdbQpeOav/view?usp=sharing)]

- CLIP Image embeddings:
    - SD ver 2.1, Torch Tensors: [[openai]()] [[laion2b]()]
    - SD ver 1.5, Torch Tensors: [[openai]()] [[laion2b]()]  


### ArtStation Data:
- SD ver 1.5: [[images in .zip](https://drive.google.com/file/d/1jn7brUSN1peBnqo3JG_LT5sBg6rPHNPm/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1Cfz2JjR8V4U5Ct_mi1uqphRzSem9gtj5/view?usp=sharing)]

- CLIP Image embeddings:
    - SD ver 1.5, Torch Tensors: [[openai](https://drive.google.com/file/d/1FCGXLnFb_lyLvqx37E8R9n5TIB7o7Wcu/view?usp=sharing)] [[laion2b](https://drive.google.com/file/d/1L8rF1pv_FjGvkvFXoECat2Qhug29XpG3/view?usp=sharing)]   
    - SD ver 2.1, Torch Tensors: [[openai](https://drive.google.com/file/d/1XvgRW-f9MVNJb7xh2yEKkLdOARaU8YFN/view?usp=sharing)] [[laion2b](https://drive.google.com/file/d/1oavqQxN7QC3X54u18pci9tEv8OMOS7eB/view?usp=sharing)] 

> Load embeddings with `torch.load('sd_2_1_openai_ViT-B-32.pt')`

## Zero-shot learning on historic images

Number of real images: 1181    
Number of generated images: 3735 (249 artists x 15 images)

### On generated images
|SD version|Image embedding|Top-1|Top-5|  
|---|---|---|---|
|2.1|openai|5.78|17.54|
||laion2b|5.49|15.1|
||laion400m|5.11|15.21|
|1.5|openai|10.15|25.46|
||laion2b|10.23|26.02|
||laion400m|8.81|22.78|



### On real images
|Image embedding|Top-1|Top-5|  
|---|---|---|
|openai|52.75|77.05|
|laion2b|55.63|79.51|
|laion400m|45.89|67.49|

## Zero-shot learning on ArtStation images

Number of real images (* filtered artists with non-ascii names): 7031      
Number of generated images: 3960 (264 artists x 15 images)     
Number of generated images (* with filtered artists): 3705 (247 artists x 15 images)


### On generated images
|SD version|Image embedding|Top-1|Top-5|  Top-1*|Top-5*|  
|---|---|---|---|---|---|
|2.1|openai|0.81|3.59|0.51|1.84|
||laion2b|0.96|3.81|0.38|1.81|
||laion400m|0.76|2.98|0.51|2.24|
|1.5|openai|0.91|3.86|0.46|2.05|
||laion2b|0.71|3.03|0.43|2.13|
||laion400m|0.66|3.11|0.57|2.24|
|Random|-|0.37|1.89|0.40|2.02|

### On real images

|Image embedding|Top-1|Top-5|  
|---|---|---|
|openai|5.85|12.22|
|laion2b|3.67|7.18|
|laion400m|2.03|6.0|

