# Information
An initial step of the project is to create a dataset/database of artists and their artworks. The dataset is `googleart_data.csv`.

We consider two types of artists to later conduct experiments with StableDiffusion and OpenCLIP models. The first type is the historical artists (i.e., very well-known artists or with a notable role in the history of art) and the second type is the so-called online artists (i.e., artists that have their artworks published online and therefore subject to be part of datasets on which generative visual models are or can be trained).

The websites or data sources considered are: GoogleArt, ArtStation.

We present below how a dataset can be created by using the scripts available in this repository. 
## **Historical artists**
### **GoogleArt**
The creation of the GoogleArt dataset consists in three phases:
- Collect artists and associated data
- Collect artwork data for artists in the artists dataset
- Save images of the artworks locally

To obtain the final dataset `googleart_art_data_final.csv` of artists and their artworks, run the scripts as follows:
```
python ./Scraping/googleart/googleart_artists_scraping.py
python ./Scraping/googleart/googleart_scraping.py
python ./Scraping/googleart/googleart_save_artworks.py
```

The file  
> googleart_artists_scraping.py

collects artists and associated data, stored in `googleart_artists.csv`.

The file
> googleart_scraping.py

is the main scraping script which collects artworks urls associated to each of the artists that are in the previously created `googleart_data.csv` file.

The file
> googleart_save_artworks.py

saves the images of the artworks locally and add additional information to the art dataset, resulting in the final dataset `googleart_data_final.csv`.

## **Online artists**
### **Artstation**
...

# Notes
- The collected data are for research purpose only. 
- It is recommended to use a VPN or proxies while scraping (at a reasonable rate) in order to avoid getting IP banned.
- In order to run the above scripts, some extensions have to be setup and present in the Scraping/ folder at chromedriver_win32/, this is the case for the browser driver (e.g., chromedriver.exe), or the code line using such extension has to be removed. An extension such as istilldontcareaboutcookies-chrome-1.1.1_0.crx is not necessary but recommended.   
- The implicit/explicit waiting times can be changed directly in the code as suits the user.