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

import argparse

import pandas as pd

import requests

month_to_number = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

def format_date(dates):
	#check for deceased
	if("Died" in dates.text):
		date_of_birth = None
		date_of_death = dates.text.split("-")[0].strip()
	elif("Born" in dates.text):
		date_of_birth = dates.text.split("-")[0].strip()
		date_of_death = None
	else:
		date_of_birth = dates.text.split("-")[0].strip()
		date_of_death = dates.text.split("-")[1].strip()
	#replace the comma
	if(date_of_birth is not None):
		date_of_birth = date_of_birth.replace(",","").split(" ")
	if(date_of_death is not None):
		date_of_death = date_of_death.replace(",","").split(" ")
	#translate the month
	if(date_of_birth is not None and len(date_of_birth[0]) == 3 ):
		date_of_birth[0] = month_to_number[date_of_birth[0]]
		date_of_birth = date_of_birth[2]+"-"+date_of_birth[0]+"-"+date_of_birth[1]
	elif date_of_birth is not None:
		#join the date of birth
		date_of_birth = '-'.join(date_of_birth)
	if(date_of_death is not None and len(date_of_death[0]) == 3 ):
		date_of_death[0] = month_to_number[date_of_death[0]]
		date_of_death = date_of_death[2]+"-"+date_of_death[0]+"-"+date_of_death[1]
	elif date_of_death is not None:
		#join the date of death
		date_of_death = '-'.join(date_of_death)
	return date_of_birth,date_of_death

def start_scraping(driver, df_artists, data_writer, nb_of_scroll_updates):

	for i, artist in df_artists.iterrows():
		artist_name = artist['artist_name']
		artist_url = artist['artist_url']
		artist_nb_artworks = artist['nb_artworks']
		print(artist_url)
		driver.get(artist_url)

		wait_driver = WebDriverWait(driver, 20)

		#wait until h2[@class='CazOhd']
		try:

			wait_driver.until(EC.presence_of_element_located((By.XPATH, ".//h2[@class='CazOhd']")))
		except TimeoutException:
			print("TimeoutException")
		try:
			dates = driver.find_element(by=By.XPATH, value=".//h2[@class='CazOhd']")
			date_of_birth,date_of_death = format_date(dates)
		except NoSuchElementException:
			date_of_birth,date_of_death = None,None

		artist_desc = driver.find_element(by=By.XPATH, value=".//div[@class='zzySAd gI3F8b']").text

		list_elements = driver.find_elements(by=By.XPATH, value=".//div[@class='z1JkWd NJ4rnc nBOv9e']")
		artist_movements_elements = list_elements[-2].find_elements(by=By.XPATH, value=".//div[@class='c2Yf5e t3RZAc lXkFp']")
		artist_mediums_elements = list_elements[-1].find_elements(by=By.XPATH, value=".//div[@class='c2Yf5e t3RZAc lXkFp']")

		#find elements with class c2Yf5e t3RZAc lXkFp
		artist_movements = []
		for movement in artist_movements_elements:
			#find element icXnic
			movement_name = movement.find_element(by=By.XPATH, value=".//h3[@class='icXnic']")
			artist_movements.append(movement_name.text)
		print(artist_movements)

		#find elements with class z1JkWd NJ4rnc nBOv9e
		artist_mediums = []
		for technique in artist_mediums_elements:
			#find element icXnic
			technique_name = technique.find_element(by=By.XPATH, value=".//h3[@class='icXnic']")
			artist_mediums.append(technique_name.text)
		print(artist_mediums)

		data_writer.writerow([artist_name, artist_nb_artworks, artist_url, date_of_birth, date_of_death,artist_desc, artist_movements,artist_mediums])

		print("Artist collected: ", artist_name)
		print("Total artists collected: ", i)

	return i


if __name__ == "__main__":

	# start time when running the script
	start_time = time.time()

	# get the driver
	driver = ga_utility.get_driver()

	# get from the command line the scraping hyper-parameters (e.g., nb_scroll_updates) for scraping
	command_line_parser = argparse.ArgumentParser()
	command_line_parser.add_argument("--nb_scroll_updates", type=int, default=15, help="Number of times to scroll down on the webpage to update it. This impacts the number of artists that will be scraped.")
	args = command_line_parser.parse_args()
	nb_scroll_updates = args.nb_scroll_updates

	#open googleart_artists.csv
	path_artists_file = "./Scraping/googleart/Data/googleart_artists.csv"
	df_artists = pd.read_csv(path_artists_file, sep="\t")

	# open file and create writer to save the data
	path_artists_file = "./Scraping/googleart/Data/googleart_artists_ext.csv"
	artists_csv_file = open(path_artists_file, 'a', encoding="utf-8")
	artists_writer = csv.writer(artists_csv_file, delimiter="\t", lineterminator="\n")

	artists_row_header = ['artist_name', 'nb_artworks', 'artist_url','date_of_birth','date_of_death', 'desc' , 'movements', 'mediums']

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
	scraping_output = start_scraping(driver, df_artists, artists_writer, nb_scroll_updates)

	print(scraping_output)

	# close csv files as nothing more to write for now
	artists_csv_file.close()

	# finish properly with the driver
	driver.close()
	driver.quit()

	# time spent for the full scraping run
	end_time = time.time()
	print(f"Finished scraping {scraping_output} artists!")
	print("Time elapsed for the scraping run: ",
	int(end_time - start_time) // 60, " minutes and ",
	int(end_time - start_time) % 60, " seconds")