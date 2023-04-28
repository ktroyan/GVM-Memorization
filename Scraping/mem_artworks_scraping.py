import mem_utility

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

	wait_driver = WebDriverWait(driver, 20)

	# get the list of selected webpage urls containing artworks to scrape
	with open("./mem_artworks_webpages.txt", "r") as file:
		artworks_urls_to_scrape = [line.rstrip() for line in file]

	nb_of_artworks_collected = 0

	index = 0

	for i, artwork_url in enumerate(artworks_urls_to_scrape):

		driver.get(artwork_url)

		time.sleep(10)

		wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//ul[@id='returns']")))
		artworks_container = driver.find_elements(by=By.XPATH, value=".//li[@class='art']")

		print(f"{len(artworks_container)} artworks in container found on page {i+1}")

		for j, artwork in enumerate(artworks_container):
			index += 1

			artwork_artist_element = artwork.find_element(by=By.XPATH, value=".//dt[@class='artist']")
			artwork_artist = artwork_artist_element.text

			artwork_title_element = artwork.find_element(by=By.XPATH, value=".//dt[@class='title']")
			artwork_title = artwork_title_element.text

			artwork_img_element = artwork.find_element(by=By.XPATH, value='.//img[@class="thumbnail"]')
			artwork_img_url = artwork_img_element.get_attribute("src")

			artwork_file_path = f"./Data/artworks/artwork_{index}.jpg"

			artwork_web_response = requests.get(artwork_img_url)
			if artwork_web_response.status_code == 200:
				with open(artwork_file_path,"wb") as file:
					file.write(artwork_web_response.content)

			print("Saving the artwork: ", index, artwork_artist, artwork_title, artwork_img_url)

			data_writer.writerow([index, artwork_artist, artwork_title, artwork_img_url])
			nb_of_artworks_collected += 1

			print("One more artwork collected! Total: ", nb_of_artworks_collected)

	return nb_of_artworks_collected


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	# get the driver
	driver = mem_utility.get_driver()

	# open file and create writer to save the data
	path_data_file = "./Data/artworks_data.csv"
	data_csv_file = open(path_data_file, 'a', encoding="utf-8")
	data_writer = csv.writer(data_csv_file, delimiter="\t", lineterminator="\n")

	data_full_row_header = ['artwork_index', 'artwork_artist', 'artwork_title', 'artwork_img_url']

	# write header of the csv file if there is no header yet
	with open(path_data_file, "r") as f:
		try:
			data_file_has_header = csv.Sniffer().has_header(f.read(1024))
		except csv.Error:  # file is empty
			data_file_has_header = False

	if not (data_file_has_header):
		# write header of the csv file
		data_writer.writerow(data_full_row_header)


	# start scraping
	scraping_output = start_scraping(driver, data_writer)

	print(scraping_output)

	# close csv files as nothing more to write for now
	data_csv_file.close()

	# finish properly with the driver
	driver.close()
	driver.quit()

	# time spent for the full scraping run
	end_time = time.time()
	print("Finished scraping TripAdvisor")
	print("Time elapsed for the scraping run: ",
	int(end_time - start_time) // 60, " minutes and ",
	int(end_time - start_time) % 60, " seconds")
