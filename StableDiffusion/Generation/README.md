# Readme

## Zero-shot learning on historic images

Number of real images: 1181    
Number of generated images: 3735 (249 artists x 15 images)

### On generated images
|SD version|Image embedding|Top-1|Top-5|  
|---|---|---|---|
|2.1|openai + ViT-B-32|4.98|14.43|
|2.1|laion2b_s34b_b79k + ViT-B-32|4.47|12.66|
|2.1|laion400m_e32 + ViT-B-32 |4.58|12.64|

### On real images
|Image embedding|Top-1|Top-5|  
|---|---|---|
|openai + ViT-B-32|52.75|77.05|
|laion2b_s34b_b79k + ViT-B-32|55.63|79.51|
|laion400m_e32 + ViT-B-32|45.89|67.49|