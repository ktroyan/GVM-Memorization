import artstation_utility as at_utility
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
import os

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
    html_element.send_keys(Keys.PAGE_DOWN)
    time.sleep(2)

def get_webpage(driver, url):
    driver.get(url)
    time.sleep(1)

def crop_image(image):
    original_dim = 691
    width=529
    height=643
    new_width = int((width/original_dim)*512)
    new_height = int((height/original_dim)*512)
    image = image.crop((0, 0, new_width, new_height))
    return image

def start_scraping(data_writer, art_df):


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
    #number_of_artists_raw = driver.find_element(by=By.XPATH, value=".//div[@class='filter-content-count text-muted']")
    #number_of_artists = int(number_of_artists_raw.split(' ')[0])

    for i in range(len(art_df)):
        # get the driver
        driver = at_utility.get_driver()
        wait_driver = WebDriverWait(driver, 20)

        if current_artist == art_df['artist_name'].iloc[i]:
            artist_count += 1
        else:
            artist_count = 0

        if artist_count > nb_of_artworks_per_artist:
            current_artist = art_df['artist_name'].iloc[i]
            continue
        
        artwork_url = art_df['artwork_url'].iloc[i]
        get_webpage(driver, artwork_url)
        
        time.sleep(2)
        
        artist_name = art_df['artist_name'].iloc[i]
        wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//time[@class='project-published']")))

        artwork_date = driver.find_element(by=By.XPATH, value=".//time[@class='project-published']").get_attribute('title')
        art_df['artwork_date'].iloc[i] = artwork_date
        print(artwork_date)

        # Given the image url, we can do the following to dl it:
        try:
            image_dl_link_element = driver.find_elements(by=By.XPATH, value=".//source")[0]
        except IndexError:
            print("No image for this artwork")
            driver.close()
            driver.quit()
            continue
        image_dl_url = str(image_dl_link_element.get_attribute("srcset"))
        art_df['image_url'].iloc[i] = image_dl_url
        print(image_dl_url)

        driver.get(image_dl_url)
        image_element = driver.find_element(by=By.XPATH, value=".//img")

        #download image
        image = Image.open(BytesIO(image_element.screenshot_as_png))

        image_path = os.path.join("./Scraping/artstation/Data/artworks/index_" + str(i) + ".png")
        #save in new file
        image.save(image_path)
        art_df['image_path'].iloc[i] = image_path
        # finish properly with the driver
        driver.close()
        driver.quit()

    return art_df


if __name__ == "__main__":

    # start time when running the script
    start_time = time.time()

    # open file and create writer to save the data
    path_art_file = "./Scraping/artstation/Data/artstation_art_data.csv"
    art_csv_file = open(path_art_file, 'a', encoding="utf-8")
    art_writer = csv.writer(art_csv_file, delimiter="\t", lineterminator="\n")

    artists_row_header = ['artist_name', 'artwork_title', 'artwork_url','artwork_cdn']

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
    art_df_final = start_scraping(art_writer, artstation_data_df)
    path_art_file = "./Scraping/artstation/Data/artstation_art_data_final.csv"
    art_df_final.to_csv(path_art_file, sep="\t", encoding="utf-8", index=False)

    print(art_df_final)

    # close csv files as nothing more to write for now
    art_csv_file.close()


    # time spent for the full scraping run
    end_time = time.time()
    print("Finished scraping ArtStation")
    print("Time elapsed for the scraping run: ",
    int(end_time - start_time) // 60, " minutes and ",
    int(end_time - start_time) % 60, " seconds")
