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


## Error analysis on generated images (historical)

### SD 1.5 with laion2b_s34b_b79k + ViT-B-32

|Artist name|Accuracy|  
|---|---|
|Pierre Bonnard|16.67%|
|Sandro Botticelli|13.33%|
|Osamu Tezuka|13.33%|
|Caspar David Friedrich|13.33%|
|Jean-Baptiste Oudry|13.33%|
|William Blake|10.0%|
|Francisco de Zurbarán|10.0%|
|Juan Gris|10.0%|
|Julia Margaret Cameron|10.0%|
|Jan Brueghel the Elder|10.0%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|A snowy wonderland with towering ice sculptures, sleighs, and a castle made of ice.|4.82%|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|4.82%|
|A romantic and whimsical scene of a couple dancing under the stars on a fairy tale-inspired balcony.|4.82%|
|A surreal cityscape with floating buildings, a rainbow bridge, and a giant clocktower in the center.|4.42%|
|A peaceful and serene lakeside scene with a dock, a canoe, and a group of friends enjoying a summer afternoon.|4.42%|
|...|...|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|1.2%|
|A tranquil beach scene with crystal-clear water, palm trees, and a couple relaxing in a hammock.|1.2%|
|A postcard-perfect Italian village with narrow streets, terracotta roofs, and a picturesque town square with a fountain.|1.2%|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.0%|
|A vibrant and colorful scene of a Carnaval parade in Rio de Janeiro with samba dancers, costumes, and floats.|0.0%|

### SD 1.5 with laion400m_e32 + ViT-B-32

|Artist name|Accuracy|  
|---|---|
|Utagawa Kuniyoshi|20.0%|
|Gustav Klimt|16.67%|
|Hokusai|16.67%|
|Alfred Sisley|16.67%|
|Camille Pissarro|13.33%|
|Lucas Cranach the Elder|13.33%|
|Pieter Bruegel the Elder|13.33%|
|Peter Paul Rubens|10.0%|
|William Blake|10.0%|
|Fernando Botero|10.0%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
||An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|4.42%|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|4.42%|
|A peaceful and serene lakeside scene with a dock, a canoe, and a group of friends enjoying a summer afternoon.|4.42%|
|A mysterious scene with an ancient temple hidden deep in a jungle, with vines and moss covering the walls.|4.42%|
|A dramatic mountain climber scene with a group of mountaineers scaling a sheer cliff face, braving wind and cold to reach the summit.|4.02%|
|...|...|
|A vibrant and colorful scene of a Carnaval parade in Rio de Janeiro with samba dancers, costumes, and floats.|1.2%|
|A whimsical and enchanting scene of a fairy tale castle in the clouds, with a rainbow in the background.|1.2%|
|A romantic and whimsical scene of a couple dancing under the stars on a fairy tale-inspired balcony.|1.2%|
|A dramatic and mystical scene of a witch performing a ritual in a dark forest, surrounded by candles and other supernatural objects.|1.2%|
|A postcard-perfect Italian village with narrow streets, terracotta roofs, and a picturesque town square with a fountain.|1.2%|


### SD 1.5 with openai + ViT-B-32

|Artist name|Accuracy|  
|---|---|
|Gordon Parks|16.67%|
|Hokusai|13.33%|
|Charles Willson Peale|13.33%|
|Henri de Toulouse-Lautrec|10.0%|
|Nicholas Roerich|10.0%|
|Odilon Redon|10.0%|
|Jean-Honoré Fragonard|10.0%|
|Claude Lorrain|10.0%|
|L. S. Lowry|10.0%|
|Emily Carr|10.0%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|A joyful and heartwarming scene of a group of children playing in a park, with swings, slides, and a merry-go-round.|6.02%|
|A romantic and whimsical scene of a couple dancing under the stars on a fairy tale-inspired balcony.|5.62%|
|A magical winter wonderland with snow-covered trees, ice skating, and a group of friends building a snowman.|4.42%|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|4.02%|
|A futuristic and eco-friendly scene of a city with massive vertical gardens covering skyscrapers, parks, and rooftop terraces.|4.02%|
|...|...|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|1.2%|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|1.2%|
|A peaceful and romantic scene of a couple sitting in a gondola, being serenaded on a moonlit canal in Venice, Italy.|0.8%|
|A vibrant and colorful scene of a Carnaval parade in Rio de Janeiro with samba dancers, costumes, and floats.|0.8%|
|A postcard-perfect Italian village with narrow streets, terracotta roofs, and a picturesque town square with a fountain.|0.4%|

### SD 2.1 with laion2b_s34b_b79k + ViT-B-32

|Artist name|Accuracy|  
|---|---|
|Eugène Atget|13.33%|
|Albert Bierstadt|13.33%|
|Peder Severin Krøyer|13.33%|
|Gustav Klimt|10.0%|
|Thomas Gainsborough|10.0%|
|Yoshitoshi|10.0%|
|Lucio Fontana|10.0%|
|Yokoyama Taikan|10.0%|
|John Singleton Copley|10.0%|
|Jan Brueghel the Elder|10.0%|

|Artist prompts|Accuracy|  
|---|---|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|4.42%|
|A romantic scene with a couple sitting on a park bench, surrounded by autumn leaves and a beautiful sunset.|4.42%|
|A mysterious scene with an ancient temple hidden deep in a jungle, with vines and moss covering the walls.|4.02%|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|3.61%|
|A tranquil space scene with a spaceship orbiting a planet, shooting stars, and a constellation of stars in the distance.|3.61%|
|...|...|
|A dark and mysterious scene of an abandoned subway station, with flickering lights and eerie echoes.|0.8%|
|A futuristic and eco-friendly scene of a city with massive vertical gardens covering skyscrapers, parks, and rooftop terraces.|0.8%|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|0.4%|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.4%|
|A whimsical and enchanting scene of a fairy tale castle in the clouds, with a rainbow in the background.|0.4%|

### SD 2.1 with laion400m_e32 + ViT-B-32

|Artist name|Accuracy|  
|---|---|
|Edgar Degas|13.33%|
|Yoshitoshi|13.33%|
|Albert Bierstadt|13.33%|
|Ramon Casas|13.33%|
|Toyohara Kunichika|13.33%|
|Paul Klee|10.0%|
|Georges Seurat|10.0%|
|Mario Testino|10.0%|
|Kunisada|10.0%|
|Salvador Dalí|10.0%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|4.02%|
|A peaceful and serene lakeside scene with a dock, a canoe, and a group of friends enjoying a summer afternoon.|4.02%|
|A mysterious scene with an ancient temple hidden deep in a jungle, with vines and moss covering the walls.|4.02%|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|3.21%|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|3.21%|
|...|...|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|0.8%|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.8%|
|A surreal cityscape with floating buildings, a rainbow bridge, and a giant clocktower in the center.|0.8%|
|A whimsical and enchanting scene of a fairy tale castle in the clouds, with a rainbow in the background.|0.8%|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.0%|

### SD 2.1 with openai + ViT-B-32

|Artist name|Accuracy|  
|---|---|
|August Macke|16.67%|
|Toyohara Chikanobu|16.67%|
|Gordon Parks|13.33%|
|Jean-Léon Gérôme|13.33%|
|Edgar Degas|10.0%|
|Utagawa Kuniyoshi|10.0%|
|Nicholas Roerich|10.0%|
|Claude Lorrain|10.0%|
|Juan Gris|10.0%|
|Alfred Stieglitz|10.0%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|A romantic scene with a couple sitting on a park bench, surrounded by autumn leaves and a beautiful sunset.|4.42%|
|A whimsical and enchanting scene of a fairy tale castle in the clouds, with a rainbow in the background.|4.02%|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|3.61%|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|3.61%|
|A mysterious scene with an ancient temple hidden deep in a jungle, with vines and moss covering the walls.|3.61%|
|...|...|
|A dramatic and mystical scene of a witch performing a ritual in a dark forest, surrounded by candles and other supernatural objects.|1.61%|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|1.2%|
|A glamorous and luxurious yacht sailing in the open sea with a beautiful sunset in the background.|1.2%|
|A postcard-perfect Italian village with narrow streets, terracotta roofs, and a picturesque town square with a fountain.|0.8%|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.4%|

## Error analysis on generated images (artstation)

### SD 1.5 with openai + ViT-B-32

|Artist name|Accuracy|
|---|---|
|Romain Jouandeau|13.33%|
|Sergey Kolesov|13.33%|
|Wenjun Lin|6.67%|
|Jakub Rozalski|6.67%|
|Ching Yeh|6.67%|
|Raphael Lacoste|6.67%|
|SIXMOREVODKA STUDIO|6.67%|
|Wadim Kashin|6.67%|
|Anton Fadeev|6.67%|
|Sebastian Luca|6.67%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|A surreal cityscape with floating buildings, a rainbow bridge, and a giant clocktower in the center.|1.52%|
|A dramatic mountain climber scene with a group of mountaineers scaling a sheer cliff face, braving wind and cold to reach the summit.|1.14%|
|A snowy wonderland with towering ice sculptures, sleighs, and a castle made of ice.|1.14%|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|0.76%|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|0.76%|
|...|...|
|A peaceful and romantic scene of a couple sitting in a gondola, being serenaded on a moonlit canal in Venice, Italy.|0.38%|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|0.38%|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|0.0%|
|A majestic and awe-inspiring scene of a safari in Africa with elephants, lions, and a beautiful sunset in the background.|0.0%|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|0.0%|


### SD 1.5 with laion2b_s34b_b79k + ViT-B-32

|Artist name|Accuracy|
|---|---|
|Stepan Alekseev|20.0%|
|Sylvain Sarrailh|13.33%|
|Alex Flores|13.33%|
|Paul Chadeisson|6.67%|
|Nivanh Chanthara|6.67%|
|Romain Jouandeau|6.67%|
|Finnian MacManus|6.67%|
|Bastien Grivet|6.67%|
|Maxim Verehin|6.67%|
|Khyzyl Saleem|6.67%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|1.14%|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|1.14%|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.76%|
|A majestic and awe-inspiring scene of a safari in Africa with elephants, lions, and a beautiful sunset in the background.|0.76%|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|0.76%|
|...|...|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|0.38%|
|A surreal cityscape with floating buildings, a rainbow bridge, and a giant clocktower in the center.|0.38%|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|0.0%|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|0.0%|
|A snowy wonderland with towering ice sculptures, sleighs, and a castle made of ice.|0.0%|

### SD 2.1 with openai + ViT-B-32

|Artist name|Accuracy|
|---|---|
|Quentin Mabille|20.0%|
|Sergey Kolesov|20.0%|
|Aleriia_V (lerapi)|13.33%|
|Sylvain Sarrailh|6.67%|
|Anato Finnstark|6.67%|
|Raf Grassetti|6.67%|
|Jakub Rozalski|6.67%|
|Andreas Rocha|6.67%|
|Raphael Lacoste|6.67%|
|Romain Jouandeau|6.67%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|A mystical forest with a majestic unicorn standing in a clearing, surrounded by glowing flowers.|1.52%|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|1.14%|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|1.14%|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|0.76%|
|A dynamic and exciting skateboard park with jumps, ramps, and half-pipes, surrounded by the city skyline.|0.76%|
|...|...|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|0.38%|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|0.38%|
|A surreal cityscape with floating buildings, a rainbow bridge, and a giant clocktower in the center.|0.38%|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|0.38%|
|A peaceful and romantic scene of a couple sitting in a gondola, being serenaded on a moonlit canal in Venice, Italy.|0.0%|


### SD 2.1 with laion2b_s34b_b79k + ViT-B-32

|Artist name|Accuracy|
|---|---|
|Bastien Grivet|13.33%|
|Sylvain Sarrailh|6.67%|
|Jama Jurabaev|6.67%|
|Ismail Inceoglu|6.67%|
|Andreas Rocha|6.67%|
|Raphael Lacoste|6.67%|
|Anna Podedworna|6.67%|
|Victor Titov|6.67%|
|Marco Plouffe (Keos Masons)|6.67%|
|Anton Fadeev|6.67%|
|...|...|

|Artist prompts|Accuracy|  
|---|---|
|An exciting and adrenaline-fueled scene of a hot air balloon race over a vast canyon.|1.14%|
|A spooky and eerie abandoned carnival scene with empty rides, broken lights, and a creepy clown lurking in the shadows.|1.14%|
|A mystical and spiritual scene of a meditating monk in a temple surrounded by cherry blossoms.|1.14%|
|A vibrant and colorful scene of a street market in Marrakech, Morocco with spices, textiles, and street performers.|0.76%|
|A romantic and picturesque scene of a couple stargazing in a field at night, with shooting stars and the Milky Way overhead.|0.76%|
|...|...|
|A peaceful and romantic scene of a couple sitting in a gondola, being serenaded on a moonlit canal in Venice, Italy.|0.38%|
|A cozy and festive holiday scene with a fireplace, stockings, and a Christmas tree decorated with lights and ornaments.|0.38%|
|A romantic and dreamy scene of a couple in a hot air balloon, floating over a picturesque countryside.|0.38%|
|A dreamy and surreal cloudscape scene with fluffy white clouds, a rainbow, and a hot air balloon floating in the distance.|0.0%|
|A snowy wonderland with towering ice sculptures, sleighs, and a castle made of ice.|0.0%|