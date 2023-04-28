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


def click_img_available_only(driver, wait_driver):

  wait_driver.until(
    EC.visibility_of_element_located((By.XPATH, ".//label[@for='img_avail']")))

  img_avail_only_element = driver.find_element(
    by=By.XPATH, value=".//label[@for='img_avail']")

  img_avail_only_element.click()


def input_artist_name(driver, wait_driver, artist_name):
  wait_driver.until(
    EC.visibility_of_element_located((By.XPATH, ".//input[@id='artist']")))

  artist_search_box_element = driver.find_element(
    by=By.XPATH, value=".//input[@id='artist']")

  artist_search_box_element.click()

  time.sleep(1)
  artist_search_box_element.send_keys(artist_name)
  time.sleep(1)
  artist_search_box_element.send_keys(Keys.RETURN)


def start_scraping(driver, data_writer, artists_writer, artists_to_scrape):

  wait_driver = WebDriverWait(driver, 20)

  nb_of_artists_collected = 0

  for artist_name in artists_to_scrape:

    driver.get("https://www.nga.gov/collection/collection-search.html")

    click_img_available_only(driver, wait_driver)

    input_artist_name(driver, wait_driver, artist_name)

    artist_url = driver.current_url

    nb_of_artists_collected += 1

    artists_writer.writerow([artist_name, artist_url])


if __name__ == "__main__":

  # start time when running the script
  start_time = time.time()

  # get the driver
  driver = mem_utility.get_driver()

  # get the list of selected scrapable cities
  with open("./artists_names.txt", "r") as file:
    artists_to_scrape = [line.rstrip() for line in file]

  # open file and create writer to save the data
  path_to_data_file = "./mem_data.csv"
  data_csv_file = open(path_to_data_file, 'a', encoding="utf-8")
  data_writer = csv.writer(data_csv_file, delimiter="\t", lineterminator="\n")

  data_full_row_header = ["artist_name", "artworks_paths"]

  # write header of the csv file if there is no header yet
  with open(path_to_data_file, "r") as f:
    try:
      data_file_has_header = csv.Sniffer().has_header(f.read(1024))
    except csv.Error:  # file is empty
      data_file_has_header = False

  if not (data_file_has_header):
    # write header of the csv file
    data_writer.writerow(data_full_row_header)

  # open file and create writer to save the artists urls
  path_to_artists_file = "./artists_urls.csv"
  artists_csv_file = open(path_to_artists_file, 'a', encoding="utf-8")
  artists_writer = csv.writer(artists_csv_file,
                              delimiter="\t",
                              lineterminator="\n")

  artists_row_header = ["artist_name", "artist_url"]
  # write header of the csv file if there is no header yet
  with open(path_to_artists_file, "r") as f:
    try:
      artists_file_has_header = csv.Sniffer().has_header(f.read(1024))
    except csv.Error:  # file is empty
      artists_file_has_header = False

  if not (artists_file_has_header):
    # write header of the csv file
    artists_writer.writerow(artists_row_header)

  # start scraping
  scraping_output = start_scraping(driver, data_writer, artists_writer,
                                   artists_to_scrape)

  print(scraping_output)

  # close csv files as nothing more to write for now
  data_csv_file.close()
  artists_csv_file.close()

  # finish properly with the driver
  driver.close()
  driver.quit()

  # clean the dataset (i.e., remove duplicates)
  path_data_file = './mem_data.csv'
  path_cleaned_dataset = './mem_data_cleaned.csv'
  clean_dataset_df = mem_utility.remove_duplicates_in_dataset(
    path_data_file, path_cleaned_dataset)

  # path_dataset_xml = './mem_data.xml'
  # mem_utility.convert_csv_to_xml(path_cleaned_dataset, path_dataset_xml)

  # time spent for the full scraping run
  end_time = time.time()
  print("Finished scraping TripAdvisor")
  print("Time elapsed for the scraping run: ",
        int(end_time - start_time) // 60, " minutes and ",
        int(end_time - start_time) % 60, " seconds")
