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
from PIL import Image
import argparse


def get_list_of_artists_data(artists_file_path):
    artists_data = []
    with open(artists_file_path, 'r', encoding='utf-8') as f:
        artists_data_str = f.read().splitlines()
        for data in artists_data_str[1:]:   # skip the header
            artists_data.append(data.split(','))
    return artists_data

def scroll_down(driver):
	html_element = driver.find_element(By.TAG_NAME, 'html')
	html_element.send_keys(Keys.PAGE_DOWN)
	time.sleep(2)

def get_webpage(driver, url):
	driver.get(url)
	time.sleep(2)

def start_scraping(data_writer,nb_of_artworks_to_collect_per_artist):
	# get the list of selected artists webpage urls containing artworks to scrape
	artists_file_path = "./Scraping/artstation/Data/artstation_artists.csv"
	artists_data = get_list_of_artists_data(artists_file_path)
	nb_of_artworks_collected = 0
	for i in range(len(artists_data)):

		driver = at_utility.get_driver()
		wait_driver = WebDriverWait(driver, 20)

		artist_name = artists_data[i][0]
		#artist_max_nb_of_artworks = artists_data[i][1]
		artist_url = artists_data[i][1]

		try :
			get_webpage(driver, artist_url)
		except:
			print("Error getting the webpage for the artist: ", artist_name)
			driver.close()
			driver.quit()
			continue
		try:
			wait_driver.until(EC.presence_of_element_located((By.XPATH, ".//div[@class='gallery']")))

			artworks_mosaic_elements = driver.find_element(by=By.XPATH, value=".//div[@class='gallery']")
		except:
			print("Error getting the artworks mosaic for the artist: ", artist_name)
			driver.close()
			driver.quit()
			continue
		artwork_elements = artworks_mosaic_elements.find_elements(by=By.XPATH, value=".//div[@class='project artist-profile']")
		nb_of_artworks = len(artwork_elements)
		for k, artwork in enumerate(artwork_elements):
			print("Collecting the artwork number: ", k+1, " for the artist: ", artist_name)
			wait_driver.until(EC.presence_of_element_located((By.XPATH, ".//a[@class='project-image']")))
			artwork_link = artwork.find_element(by=By.XPATH, value=".//a[@class='project-image']")
			artwork_url = artwork_link.get_attribute("href")
			#contains image and something else in class
			wait_driver.until(EC.presence_of_element_located((By.XPATH, ".//img[contains(@class,'image')]")))
			
			artwork_image = artwork_link.find_element(by=By.XPATH, value=".//img[contains(@class,'image')]")
			artwork_title = artwork_image.get_attribute("alt")
			

			print("Saving the artwork: ", artwork_title, " by ", artist_name)
			data_writer.writerow([artist_name, artwork_title, artwork_url])
			
			nb_of_artworks_collected += 1

			print("Total artworks collected: ", nb_of_artworks_collected)

			if(k%10==0):
				print("Scrolling down")
				scroll_down(driver)

			if(k>=nb_of_artworks_to_collect_per_artist-1):
				break
		driver.close()
		driver.quit()
	return nb_of_artworks_collected


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	#command line arg for the number of picture to take
	command_line_parser = argparse.ArgumentParser()

	command_line_parser.add_argument("--nb_artwork_per_artist", type=int, default=10, help="Number of artwork per artists that will be scraped.")
	args = command_line_parser.parse_args()

	artwork_per_artist = args.nb_artwork_per_artist

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


	# start scraping
	scraping_output = start_scraping(art_writer,artwork_per_artist)

	print(scraping_output)

	# close csv files as nothing more to write for now
	art_csv_file.close()

	# finish properly with the driver
	#driver.close()
	#driver.quit()

	# time spent for the full scraping run
	end_time = time.time()
	print("Finished scraping ArtStation")
	print("Time elapsed for the scraping run: ",
	int(end_time - start_time) // 60, " minutes and ",
	int(end_time - start_time) % 60, " seconds")
