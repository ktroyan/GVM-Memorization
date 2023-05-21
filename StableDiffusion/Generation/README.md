# Readme

## Real Images

### Historic Data

- [[images in .zip](https://drive.google.com/file/d/1ZAegLoQ20XXI1tWDgOZFVJdEraMW6Bqd/view?usp=sharing)] [[image descriptions](https://drive.google.com/file/d/1ZX1GpSbKcLjbdA48TNVdEIPQ-FgaldUr/view?usp=sharing)]
- CLIP Image embeddings:
    - [[openai](https://drive.google.com/file/d/1HWUpYLGZErlhaahLladfFj0YuebhjY9s/view?usp=sharing)] [[laion2b](https://drive.google.com/file/d/10ZnfiCjCw-ELPcWCiU0sZO0lwmKdyOtH/view?usp=sharing)]

### ArtStation Data



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


## Error analysis on generated images (historical)

### SD 1.5 with laion2b_s34b_b79k + ViT-B-32

|Artist name|Probability|  
|---|---|
|Albert Bierstadt|0.056|
|Canaletto|0.047|
|Frederic Edwin Church|0.043|
|Jan Brueghel the Elder|0.029|
|Hiroshige|0.028|
|...|...|
|Lovis Corinth|7.650e-05|
|Hans Holbein the Younger|7.476e-05|
|Yue Minjun|6.187e-05|
|Gian Lorenzo Bernini|3.952e-05|
|Hendrick Goltzius|2.401e-05|

|Artist prompts|Probability|  
|---|---|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|0.988|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|0.940|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|0.886|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|0.862|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|0.643|
|...|...|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|0.111|
|A peaceful and romantic scene of a couple sitting in a gondola, being serenaded on a moonlit canal in Venice, Italy.|0.106|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|0.037|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|0.0035|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.0001|


### SD 1.5 with openai + ViT-B-32

|Artist name|Probability|  
|---|---|
|Canaletto|0.030|
|George Stubbs|0.028|
|Jean-Léon Gérôme|0.024|
|Alessandro Mendini|0.022|
|Claude Lorrain|0.022|
|...|...|
|Martin Schongauer|0.00031|
|Utamaro|0.00030|
|Jusepe de Ribera|0.00024|
|Jacob Jordaens|0.00020|
|Lucio Fontana|0.00018|

|Artist prompts|Probability|  
|---|---|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|0.7206|
|A dramatic mountain climber scene with a group of mountaineers scaling a sheer cliff face, braving wind and cold to reach the summit.|0.540|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|0.4994|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|0.454|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.349|
|...|...|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|0.129|
|A majestic and awe-inspiring scene of a safari in Africa with elephants, lions, and a beautiful sunset in the background.|0.129|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|0.128|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|0.020|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.0081|


### SD 2.1 with laion2b_s34b_b79k + ViT-B-32

|Artist name|Probability|  
|---|---|
|Santiago Rusiñol|0.079|
|Karl Friedrich Schinkel|0.073|
|Henri Rousseau|0.059|
|Fernando Botero|0.046|
|Giorgio de Chirico|0.041|
|...|...|
|Lovis Corinth|1.90e-05|
|Paul Cézanne|1.35e-05|
|Julia Margaret Cameron|1.29e-05|
|Anthony van Dyck|1.21e-05|
|Jacob Jordaens|8.86e-05|

|Artist prompts|Probability|  
|---|---|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|0.907|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|0.793|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.690|
|A dramatic mountain climber scene with a group of mountaineers scaling a sheer cliff face, braving wind and cold to reach the summit.|0.441|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|0.179|
|...|...|
|A surreal cityscape with floating buildings, a rainbow bridge, and a giant clocktower in the center.|0.009|
|A majestic and awe-inspiring scene of a safari in Africa with elephants, lions, and a beautiful sunset in the background.|0.004|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.003|
|A snowy wonderland with towering ice sculptures, sleighs, and a castle made of ice.|0.001|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|9.18e-05|


### SD 2.1 with openai + ViT-B-32

|Artist name|Probability|  
|---|---|
|Henri Rousseau|0.058|
|Kenny Scharf|0.026|
|Alessandro Mendini|0.024|
|Salvador Dalí|0.022|
|Georgia O'Keeffe|0.019|
|...|...|
|Jacob Jordaens|0.00012|
|Jusepe de Ribera|0.00012|
|Martin Schongauer|8.87e-05|
|Félicien Rops|8.16e-05|
|Hendrick Goltzius|7.01e-05|

|Artist prompts|Probability|  
|---|---|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|0.513|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.286|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|0.263|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|0.200|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|0.174|
|...|...|
|A majestic and awe-inspiring scene of a safari in Africa with elephants, lions, and a beautiful sunset in the background.|0.021|
|A snowy wonderland with towering ice sculptures, sleighs, and a castle made of ice.|0.016|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|0.015|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|0.0079|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.0073|


