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
import PIL
from PIL import Image
from io import StringIO



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
	time.sleep(2)

def start_scraping(driver, data_writer):
	
	# get the list of selected artists webpage urls containing artworks to scrape
	artists_file_path = "./Scraping/googleart/Data/googleart_artists.csv"
	artists_data = get_list_of_artists_data(artists_file_path)

	wait_driver = WebDriverWait(driver, 20)

	nb_of_artworks_collected = 0

	for i in range(len(artists_data)):
		artist_name = artists_data[i][0]
		artist_max_nb_of_artworks = artists_data[i][1]
		artist_url = artists_data[i][2]

		get_webpage(driver, artist_url)

		scroll_down(driver)

		# TODO: IF needed, scroll laterally right to get more artworks of the artist on the webpage
		# ...

		artworks_mosaic_elements = driver.find_elements(by=By.XPATH, value=".//div[@class='wcg9yf']")

		for j, artworks_mosaic in enumerate(artworks_mosaic_elements):
			
			artwork_elements = artworks_mosaic.find_elements(by=By.XPATH, value=".//a[@class='e0WtYb kdYEFe ZEnmnd PJLMUc']")
			
			for k, artwork in enumerate(artwork_elements):

				artwork_title = artwork.get_attribute("title")
				artwork_url = artwork.get_attribute("href")

				print("Saving the artwork: ", artwork_title, " by ", artist_name)
				data_writer.writerow([artist_name, artwork_title, artwork_url])
				
				nb_of_artworks_collected += 1

				print("Total artworks collected: ", nb_of_artworks_collected)

	return nb_of_artworks_collected


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	# get the driver
	driver = ga_utility.get_driver()

	# open file and create writer to save the data
	path_art_file = "./Scraping/googleart/Data/googleart_art_data.csv"
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


	# start scraping
	scraping_output = start_scraping(driver, art_writer)

	print(scraping_output)

	# close csv files as nothing more to write for now
	art_csv_file.close()

	# finish properly with the driver
	driver.close()
	driver.quit()

	# time spent for the full scraping run
	end_time = time.time()
	print("Finished scraping TripAdvisor")
	print("Time elapsed for the scraping run: ",
	int(end_time - start_time) // 60, " minutes and ",
	int(end_time - start_time) % 60, " seconds")
