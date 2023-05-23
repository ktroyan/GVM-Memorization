import googleart_utility as ga_utility

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import InvalidArgumentException, NoSuchElementException, ElementClickInterceptedException, TimeoutException, NoSuchWindowException, StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import time
import csv

import requests
from PIL import Image
from io import StringIO, BytesIO
import os
import pandas as pd
import urllib
import base64

def get_webpage(driver, url):
    driver.get(url)
    time.sleep(3)

# taken from SO. TODO: give credits
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

def crop_image(image, original_dims, new_dims, intrisic_dims):
    new_width = int((new_dims[0]/original_dims[0])*intrisic_dims[0])
    new_height = int((new_dims[1]/original_dims[1])*intrisic_dims[1])
    image = image.crop((0, 0, new_width, new_height))
    return image

def start_scraping(driver, data_writer, art_df):

    wait_driver = WebDriverWait(driver, 10)

    # as a check, we make sure that all the samples considered have an artwork url
    art_df = art_df[art_df['artwork_url'].notna()]
    
    # add columns to the dataframe
    art_df['artwork_date'] = ""
    art_df['image_url'] = ""
    art_df['image_path'] = ""

    # number of artworks to download per artist
    nb_of_artworks_per_artist = 20

    previous_artist = ""
    artist_count = 0
    index = 0

    for i in range(len(art_df)):
        print("Current artist is: ", art_df['artist_name'].iloc[i])

        if previous_artist == art_df['artist_name'].iloc[i]:
            print("Artist is the same as for the previous artwork.")
            artist_count += 1
        else:
            previous_artist = art_df['artist_name'].iloc[i]
            print("Artwork of a different artist found. Resetting the \"number of artwork per artist\" counter.")
            artist_count = 0

        if artist_count >= nb_of_artworks_per_artist:
            print("Already saved enough artworks for this artist. Skipping this artwork.")
            previous_artist = art_df['artist_name'].iloc[i]
            continue

        artwork_url = art_df['artwork_url'].iloc[i]
        try:
            get_webpage(driver, artwork_url)
        except InvalidArgumentException:
            print("Invalid URL. Skipping to the next artwork.")
            continue

        # Save the date of the artwork, just in case
        try:
            # wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//span[@class='QtzOu']")))
            artwork_date = driver.find_element(by=By.XPATH, value=".//span[@class='QtzOu']")
            art_df['artwork_date'].iloc[i] = artwork_date.text
            print(artwork_date.text)
        except NoSuchElementException:
            print("No artwork date found. Skipping to the next artwork.")
            continue

        # Given the image url, we can do the following to dl it:
        try:
            image_dl_link_element = driver.find_element(by=By.XPATH, value=".//img[@class='XkWAb-LmsqOc XkWAb-Iak2Lc']")
            image_dl_url = str(image_dl_link_element.get_attribute("src"))
            art_df['image_url'].iloc[i] = image_dl_url
        except NoSuchElementException:
            print("No image link found. Skipping to the next artwork.")
            continue
        
        # Get image from blob bytes
        bytes_from_blob = get_file_content_chrome(driver, image_dl_url)
        image = Image.open(BytesIO(bytes_from_blob))

        # Get the intrisic dimension (number of pixels) of the image
        try:
            image_original_dims_element = driver.find_element(by=By.XPATH, value=".//div[@class='jyCOCf WtuUG']/div[@class='MJA7d DgsXq tpfJYe pyGVEf']/div[@class='ScDhKc']/div[@class='XkWAb-GfpNfc']/div[@class='XkWAb-cYRDff']/img[@class='XkWAb-LmsqOc XkWAb-Iak2Lc']")
            original_width = int(image_original_dims_element.get_attribute("style").split("width: ")[1].split("px")[0])
            original_height = int(image_original_dims_element.get_attribute("style").split("height: ")[1].split("px")[0])
            original_dims = (original_width, original_height)
            # print(original_dims)
        except NoSuchElementException:
            print("No original dimensions found. Skipping to the next artwork.")
            continue

        # Get the new dimension (number of pixels) of the image
        try:
            image_dims_element = driver.find_element(by=By.XPATH, value=".//div[@class='jyCOCf WtuUG']/div[@class='MJA7d DgsXq tpfJYe pyGVEf']/div[@class='ScDhKc']/div[@class='XkWAb-GfpNfc']/div[@class='XkWAb-cYRDff']")
            new_width = int(image_dims_element.get_attribute("style").split("width: ")[1].split("px")[0])
            new_height = int(image_dims_element.get_attribute("style").split("height: ")[1].split("px")[0])
            new_dims = (new_width, new_height)
            # print(new_dims)
        except NoSuchElementException:
            print("No new dimensions found. Skipping to the next artwork.")
            continue

        # Get the intrisic dimension (number of pixels) of the image
        intrisic_dims = (512, 512)  # hard-coded as (apparently) not available from the webpage
        
        image = crop_image(image, original_dims, new_dims, intrisic_dims)

        index += 1
        image_path = f'./Scraping/googleart/Data/artworks_images/image{index}.jpg'
        
        image.save(image_path)

        art_df['image_path'].iloc[i] = image_path

        print(f"Saved artwork {index} called \"{art_df['artwork_title'].iloc[i]}\" from {art_df['artist_name'].iloc[i]}")
        print("Image path: ", image_path)

        # Save the data sample to the csv file
        data_writer.writerow([art_df['artist_name'].iloc[i], art_df['artwork_title'].iloc[i], art_df['artwork_date'].iloc[i], art_df['artwork_url'].iloc[i], art_df['image_url'].iloc[i], art_df['image_path'].iloc[i]])

    return art_df


if __name__ == "__main__":

    # start time when running the script
    start_time = time.time()

    # get the driver
    driver = ga_utility.get_driver()

    path_art_file = "./Scraping/googleart/Data/googleart_art_data.csv"
    path_final_art_file = "./Scraping/googleart/Data/googleart_art_data_final.csv"

    # open google_art_data.csv file in pandas dataframe
    googleart_data_df = pd.read_csv(path_art_file, sep='\t', encoding='utf-8')
    #filter artist that have less than 20 artworks
    googleart_data_df = googleart_data_df.groupby('artist_name').filter(lambda x: len(x) > 20)
    #print the number of artists
    print("Number of artists: ", len(googleart_data_df.groupby('artist_name').groups.keys()))

    # open file and create writer to save the data
    art_csv_file = open(path_final_art_file, 'a', encoding="utf-8")
    art_writer = csv.writer(art_csv_file, delimiter="\t", lineterminator="\n")

    artists_row_header = ['artist_name', 'artwork_title', 'artwork_date', 'artwork_url', 'image_url', 'image_path']

    # write header of the csv file if there is no header yet
    with open(path_final_art_file, "r") as f:
        try:
            data_file_has_header = csv.Sniffer().has_header(f.read(1024))
        except csv.Error:  # file is empty
            data_file_has_header = False

    if not (data_file_has_header):
        # write header of the csv file
        art_writer.writerow(artists_row_header)


    # start scraping
    art_df_final = start_scraping(driver, art_writer, googleart_data_df)
    # art_df_final.to_csv(path_final_art_file, sep="\t", encoding="utf-8", index=False) # uncomment this if we want a dataset with all the artists even though scraping is incomplete 

    print(art_df_final)

    # close csv files as nothing more to write for now
    art_csv_file.close()

    # finish properly with the driver
    driver.close()
    driver.quit()

    # time spent for the full scraping run
    end_time = time.time()
    print("Time elapsed for the scraping run: ",
    int(end_time - start_time) // 60, " minutes and ",
    int(end_time - start_time) % 60, " seconds")
