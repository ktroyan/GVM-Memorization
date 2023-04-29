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

def start_scraping(driver, data_writer):
    googleart_artists_webpage = "https://artsandculture.google.com/u/0/category/artist?tab=pop"
	
    driver.get(googleart_artists_webpage)

    wait_driver = WebDriverWait(driver, 20)

    nb_of_artists_collected = 0

    # scroll down the webpage several times to update the artists present on the webpage
    nb_of_scrolls = 10
    for _ in range(nb_of_scrolls):
        html_element = driver.find_element(By.TAG_NAME, 'html')
        html_element.send_keys(Keys.END)
        time.sleep(2)

    artist_elements = driver.find_elements(by=By.XPATH, value=".//li[@class='DuHQbc']")

    for i, artist in enumerate(artist_elements):

        artist_element = artist.find_element(by=By.XPATH, value=".//a[@class='e0WtYb HpzMff PJLMUc']")
        artist_name = artist_element.text.split("\n")[0]
        artist_nb_artworks = int(artist_element.text.split("\n")[1].split(" ")[0].replace(",", ""))
        artist_url = artist_element.get_attribute("href")
        
        print("Saving the artist: ", artist_name)
        print("Url: ", artist_url)

        data_writer.writerow([artist_name, artist_nb_artworks, artist_url])
        
        nb_of_artists_collected += 1
    
        print("Total artists collected: ", nb_of_artists_collected)

    return nb_of_artists_collected


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	# get the driver
	driver = ga_utility.get_driver()

	# open file and create writer to save the data
	path_artists_file = "./Scraping/googleart/Data/googleart_artists.csv"
	artists_csv_file = open(path_artists_file, 'a', encoding="utf-8")
	artists_writer = csv.writer(artists_csv_file, delimiter="\t", lineterminator="\n")

	artists_row_header = ['artist_name', 'nb_artworks', 'artist_url']

	# write header of the csv file if there is no header yet
	with open(path_artists_file, "r") as f:
		try:
			data_file_has_header = csv.Sniffer().has_header(f.read(1024))
		except csv.Error:  # file is empty
			data_file_has_header = False

	if not (data_file_has_header):
		# write header of the csv file
		artists_writer.writerow(artists_row_header)


	# start scraping
	scraping_output = start_scraping(driver, artists_writer)

	print(scraping_output)

	# close csv files as nothing more to write for now
	artists_csv_file.close()

	# finish properly with the driver
	driver.close()
	driver.quit()

	# time spent for the full scraping run
	end_time = time.time()
	print("Finished scraping TripAdvisor")
	print("Time elapsed for the scraping run: ",
	int(end_time - start_time) // 60, " minutes and ",
	int(end_time - start_time) % 60, " seconds")
