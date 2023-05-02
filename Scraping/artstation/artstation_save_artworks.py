import googleart_utility as ga_utility

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException, NoSuchWindowException, StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import time
import csv

import requests
from PIL import Image
from io import StringIO, BytesIO

import pandas as pd
import urllib
import base64

def get_list_of_artists_data(artists_file_path):
    artists_data = []
    with open(artists_file_path, 'r', encoding='utf-8') as f:
        artists_data_str = f.read().splitlines()
        for data in artists_data_str[1:]:   # skip the header
            artists_data.append(data.split('\t'))
    return artists_data

def scroll_down(driver):
    html_element = driver.find_element(By.TAG_NAME, 'html')
    html_element.send_keys(Keys.END)
    time.sleep(2)

def get_webpage(driver, url):
    driver.get(url)
    time.sleep(1)

def get_file_content_chrome(driver, uri):
  result = driver.execute_async_script("""
    var uri = arguments[0];
    var callback = arguments[1];
    var toBase64 = function(buffer){for(var r,n=new Uint8Array(buffer),t=n.length,a=new Uint8Array(4*Math.ceil(t/3)),i=new Uint8Array(64),o=0,c=0;64>c;++c)i[c]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charCodeAt(c);for(c=0;t-t%3>c;c+=3,o+=4)r=n[c]<<16|n[c+1]<<8|n[c+2],a[o]=i[r>>18],a[o+1]=i[r>>12&63],a[o+2]=i[r>>6&63],a[o+3]=i[63&r];return t%3===1?(r=n[t-1],a[o]=i[r>>2],a[o+1]=i[r<<4&63],a[o+2]=61,a[o+3]=61):t%3===2&&(r=(n[t-2]<<8)+n[t-1],a[o]=i[r>>10],a[o+1]=i[r>>4&63],a[o+2]=i[r<<2&63],a[o+3]=61),new TextDecoder("ascii").decode(a)};
    var xhr = new XMLHttpRequest();
    xhr.responseType = 'arraybuffer';
    xhr.onload = function(){ callback(toBase64(xhr.response)) };
    xhr.onerror = function(){ callback(xhr.status) };
    xhr.open('GET', uri);
    xhr.send();
    """, uri)
  if type(result) == int :
    raise Exception("Request failed with status %s" % result)
  return base64.b64decode(result)

def crop_image(image):
    original_dim = 691
    width=529
    height=643
    new_width = int((width/original_dim)*512)
    new_height = int((height/original_dim)*512)
    image = image.crop((0, 0, new_width, new_height))
    return image

def start_scraping(driver, data_writer, art_df):

    wait_driver = WebDriverWait(driver, 20)

    nb_of_artworks_downloaded = 0

    # as a check, we make sure that all the sample considered have an artwork url
    art_df = art_df[art_df['artwork_url'].notna()]
    
    # add columns to the dataframe
    art_df['artwork_date'] = ""
    art_df['image_url'] = ""
    art_df['image_path'] = ""

    nb_of_artworks_per_artist = 5

    current_artist = ""
    artist_count = 0
    index = 0
    
    number_of_artists_raw = driver.find_element(by=By.XPATH, value=".//div[@class='filter-content-count text-muted']")
    number_of_artists = int(number_of_artists_raw.split(' ')[0])

    for i in range(len(art_df)):

        if current_artist == art_df['artist_name'].iloc[i]:
            artist_count += 1
        else:
            artist_count = 0

        if artist_count > nb_of_artworks_per_artist:
            current_artist = art_df['artist_name'].iloc[i]
            continue

        artwork_url = art_df['artwork_url'].iloc[i]
        get_webpage(driver, artwork_url)

        wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//li[@_ngcontent-snh-c98]")))
    
        artist_name = driver.find_element(by=By.XPATH, value=".//a[@class='text-white']")

        artwork_date = driver.find_element(by=By.XPATH, value=".//span[@class='QtzOu']")
        art_df['artwork_date'].iloc[i] = artwork_date.text
        print(artwork_date.text)

        # Given the image url, we can do the following to dl it:
        image_dl_link_element = driver.find_element(by=By.XPATH, value=".//img[@class='XkWAb-LmsqOc XkWAb-Iak2Lc']")
        image_dl_url = str(image_dl_link_element.get_attribute("src"))
        art_df['image_url'].iloc[i] = image_dl_url
        print(image_dl_url)


        index += 1
        image_path = f'./Scraping/artstation/Data/image{index}.jpg'

        bytes_from_blob = get_file_content_chrome(driver, image_dl_url)
        print(bytes_from_blob)
        
        # get image from bytes
        image = Image.open(BytesIO(bytes_from_blob))
        #image = crop_image(image)

        # with=529px, height=643px. Original: 691x691
        # Get the intrisic dimension (number of pixels) of the image
        
        
        image_intrisic_dims_element = driver.find_element(by=By.XPATH, value=".//img[@class='XkWAb-LmsqOc XkWAb-Iak2Lc']")
        instrisic_width = image_intrisic_dims_element.get_atribute("width")
        instrisic_height = image_intrisic_dims_element.get_atribute("height")

        image_dims_element = driver.find_element(by=By.XPATH, value=".//img[@class='XkWAb-cYRDff']")
        image_dims_element.get_atribute("")

        image = crop_image(image, )
        
        image.save(image_path)

        art_df['image_path'].iloc[i] = image_path

    return art_df


if __name__ == "__main__":

    # start time when running the script
    start_time = time.time()

    # get the driver
    driver = ga_utility.get_driver()

    # open file and create writer to save the data
    path_art_file = "./Scraping/artstation/Data/artstation_art_data.csv"
    art_csv_file = open(path_art_file, 'a', encoding="utf-8")
    art_writer = csv.writer(art_csv_file, delimiter="\t", lineterminator="\n")

    artists_row_header = ['artist_name', 'artwork_title', 'artwork_url']

    # write header of the csv file if there is no header yet
    with open(path_art_file, "r") as f:
        try:
            data_file_has_header = csv.Sniffer().has_header(f.read(1024))
        except csv.Error:  # file is empty
            data_file_has_header = False

    if not (data_file_has_header):
        # write header of the csv file
        art_writer.writerow(artists_row_header)

    # open artstation_art_data.csv file in pandas dataframe
    artstation_data_df = pd.read_csv(path_art_file, sep='\t', encoding='utf-8')

    # start scraping
    art_df_final = start_scraping(driver, art_writer, artstation_data_df)
    path_art_file = "./Scraping/artstation/Data/artstation_art_data_final.csv"
    art_df_final.to_csv(path_art_file, sep="\t", encoding="utf-8", index=False)

    print(art_df_final)

    # close csv files as nothing more to write for now
    art_csv_file.close()

    # finish properly with the driver
    driver.close()
    driver.quit()

    # time spent for the full scraping run
    end_time = time.time()
    print("Finished scraping ArtStation")
    print("Time elapsed for the scraping run: ",
    int(end_time - start_time) // 60, " minutes and ",
    int(end_time - start_time) % 60, " seconds")
