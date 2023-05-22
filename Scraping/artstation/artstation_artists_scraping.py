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

def scroll_down_webpage(driver, nb_scroll_updates):
	# for i in range(nb_scroll_updates):
		# html_element = driver.find_element(By.TAG_NAME, 'html')
		# html_element.send_keys(Keys.END)
		# html_element.send_keys(Keys.PAGE_DOWN)

		# driver.execute_script("window.scrollBy(0, 250)")
		# driver.execute_script("window.scrollTo(0, window.scrollY + 200)")
		# time.sleep(2)

	scroll_element = driver.find_element(By.ID, "fb-root")
	driver.execute_script("arguments[0].scrollIntoView();", scroll_element)
	# driver.execute_script('window.scrollTo(0, document.getElementById("fb-root").scrollHeight);')
	print("scrolling")

def start_scraping(driver, data_writer, nb_scroll_updates, min_nb_followers):

	wait_driver = WebDriverWait(driver, 20)

	# go to the webpage that lists the artists on ArtStation
	googleart_artists_webpage = "https://www.artstation.com/search/artists?sort_by=followers&followers_count_more_than=" + str(min_nb_followers)
	
	try:
		driver.get(googleart_artists_webpage)
		time.sleep(5)	# wait until the page has loaded
		wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='filter-content-count text-muted']")))
	except TimeoutException:
		print("TimeoutException: The webpage did not load in time.")
		return

	
	grid_view_button = driver.find_element(by=By.XPATH, value=".//button[@class='btn btn-default active']")
	time.sleep(2)
	grid_view_button.click()
	time.sleep(2)

	# scroll down the webpage several times to update the number of artists present on the webpage
	scroll_down_webpage(driver, nb_scroll_updates)

	artist_elements = driver.find_elements(by=By.XPATH, value=".//div[@class='artists-list-info flex-shrink-0']")

	nb_of_artists_collected = 0

	# iterate over the artists in the webpage and collect their info and profile url
	for i, artist in enumerate(artist_elements):
		xpath_artist_info = ".//div[@class='d-flex align-items-start']/div[@class='artists-list-card']/div[@class='artists-list-name']/a[@class='text-white']"
		artist_profile = artist.find_element(by=By.XPATH, value=xpath_artist_info)
		artist_name = artist_profile.text
		artist_url = artist_profile.get_attribute("href")
		
		print("Saving the artist: ", artist_name)
		print("Url: ", artist_url)

		data_writer.writerow([artist_name, artist_url])
		
		nb_of_artists_collected += 1

		print("Total artists collected: ", nb_of_artists_collected)

	print(f"Collected {nb_of_artists_collected} artists.")

	return nb_of_artists_collected


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	# get from the command line the scraping hyper-parameters (e.g., nb_scroll_updates) for scraping
	command_line_parser = argparse.ArgumentParser()

	command_line_parser.add_argument("--nb_scroll_updates", type=int, default=30, help="Number of times to scroll down on the webpage to update it. This impacts the number of artists that will be scraped.")
	
	# set the number of followers filter that selects artists that have at least artist_n_followers followers; we are then more likely to have "famous" online artists that exist in training sets of widely used StaleDiffusion models
	command_line_parser.add_argument("--min_nb_followers", type=int, default=10000, help="Number of followers that an artist must have to be scraped. This impacts the number of artists that will be scraped.")	

	args = command_line_parser.parse_args()

	nb_scroll_updates = args.nb_scroll_updates
	min_nb_followers = args.min_nb_followers

	# get the driver
	driver = at_utility.get_driver()

	# open file and create writer to save the data
	path_artists_file = "./Scraping/artstation/Data/artstation_artists.csv"
	artists_csv_file = open(path_artists_file, 'a', encoding="utf-8")
	artists_writer = csv.writer(artists_csv_file, delimiter="\t", lineterminator="\n")

	artists_row_header = ['artist_name', 'artist_url']

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
	scraping_output = start_scraping(driver, artists_writer, nb_scroll_updates, min_nb_followers)

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
