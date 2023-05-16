# Readme


## Generated Images

- Historic Data
    - SD ver 2.1: [[images in .zip](https://drive.google.com/file/d/1d4zHZ0sBdjSomjjS137DTP-yhEAWJ4Vh/view?usp=sharing)]  [[images stored in .npy](https://drive.google.com/file/d/1hVLcaKpu-CuVEZB7Kq9QGfUfCqrhT8yt/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1_Y0WiH7Pac_5OOyzkALoaG8NonAedipf/view?usp=sharing)]
    - SD ver 1.5: [[images in .zip](https://drive.google.com/file/d/1ceQyy9kAdUHq1HGbcaW1fsLFi9yQPTWz/view?usp=share_link)] [[image descriptions](https://drive.google.com/file/d/1VXuwaNIYVt1JFUIh6A5p65AIHFM0HHzJ/view?usp=sharing)]


## Zero-shot learning on historic images

Number of real images: 1181    
Number of generated images: 3735 (249 artists x 15 images)

### On generated images
|SD version|Image embedding|Top-1|Top-5|  
|---|---|---|---|
|2.1|openai + ViT-B-32|5.78|17.54|
|2.1|laion2b_s34b_b79k + ViT-B-32|4.47|12.66|
|2.1|laion400m_e32 + ViT-B-32 |4.58|12.64|
|1.5|openai + ViT-B-32|10.15|25.46|
|1.5|laion2b_s34b_b79k + ViT-B-32|10.23|26.02|
|1.5|laion400m_e32 + ViT-B-32 |8.81|22.78|



### On real images
|Image embedding|Top-1|Top-5|  
|---|---|---|
|openai + ViT-B-32|52.75|77.05|
|laion2b_s34b_b79k + ViT-B-32|55.63|79.51|
|laion400m_e32 + ViT-B-32|45.89|67.49|
