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

import requests

import argparse

# def go_to_search_artists_page(driver, wait_driver):
# 	wait_driver.until(
#     EC.visibility_of_element_located((By.XPATH, ".//input[@class='autocomplete-input main-menu-search-input']")))
# 	search_box_element = driver.find_element(
#     by=By.XPATH, value=".//input[@class='autocomplete-input main-menu-search-input']")

# 	search_box_element.click()

# 	time.sleep(1)

# 	artist_search_box_element = driver.find_element(
#     by=By.XPATH, value=".//input[@href='/search/artists']")

# 	artist_search_box_element.click()

# 	time.sleep(1)

# def add_followers_filter(driver, wait_driver, n_followers):
# 	add_filter_box_element = driver.find_element(
# 	by=By.XPATH, value=".//button[text()='Click me']")

# 	add_filter_box_element.click()

# 	time.sleep(1)

# 	follower_box_element = driver.find_element(
# 	by=By.XPATH, value=".//li[@id='select2-zevd-result-jf1s-followers_count']")

# 	follower_box_element.click()

# 	time.sleep(1)

# 	more_than_box_element = driver.find_element(
# 	by=By.XPATH, value=".//li[@id='select2-3rs3-result-6kti-more_than']")

# 	more_than_box_element.click()

# 	time.sleep(1)

# 	follower_input_field = driver.find_element(
# 	by=By.XPATH, value=".//input[@id='form-control search-filters-control-input ng-pristine ng-valid ng-touched']")

# 	follower_input_field.send_keys(n_followers)

# 	time.sleep(1)



def start_scraping(driver, data_writer):

	artist_n_followers = 10000

	googleart_artists_webpage = "https://www.artstation.com/search/artists?sort_by=followers&followers_count_more_than=" + str(artist_n_followers)

	driver.get(googleart_artists_webpage)

	wait_driver = WebDriverWait(driver, 20)

	time.sleep(5)

	wait_driver.until(
    EC.visibility_of_element_located((By.XPATH, ".//div[@class='filter-content-count text-muted']")))

	number_of_artists_raw = driver.find_element(by=By.XPATH, value=".//div[@class='filter-content-count text-muted']")




	number_of_artists = int(number_of_artists_raw.split(' ')[0].replace(',',''))

	nb_of_artists_collected = 0

	# scroll down the webpage several times to update the artists present on the webpage
	'''
	nb_of_scrolls = 10
	for _ in range(nb_of_scrolls):
		html_element = driver.find_element(By.TAG_NAME, 'html')
		html_element.send_keys(Keys.END)
		time.sleep(2)
	'''
	artist_elements = driver.find_elements(by=By.XPATH, value=".//li[_ngcontent-bil-c98]")

	for i, artist in enumerate(artist_elements):

		artist_name = artist.find_element(by=By.XPATH, value=".//a[@class='text-white']")
		artist_url = artist_name.get_attribute("href")
		
		print("Saving the artist: ", artist_name)
		print("Url: ", artist_url)

		data_writer.writerow([artist_name, artist_url])
		
		nb_of_artists_collected += 1

		print("Total artists collected: ", nb_of_artists_collected)

	return nb_of_artists_collected


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	# get from the command line the scraping hyper-parameters (e.g., nb_scroll_updates) for scraping
	command_line_parser = argparse.ArgumentParser()
	command_line_parser.add_argument("--nb_scroll_updates", type=int, default=15, help="Number of times to scroll down on the webpage to update it. This impacts the number of artists that will be scraped.")
	args = command_line_parser.parse_args()
	nb_scroll_updates = args.nb_scroll_updates

	# get the driver
	driver = at_utility.get_driver()

	# open file and create writer to save the data
	path_artists_file = "./Scraping/artstation/Data/artstation_artists.csv"
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
	print("Finished scraping ArtStation")
	print("Time elapsed for the scraping run: ",
	int(end_time - start_time) // 60, " minutes and ",
	int(end_time - start_time) % 60, " seconds")
