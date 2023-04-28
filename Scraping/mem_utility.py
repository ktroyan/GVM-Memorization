from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC

import random
import string

import pandas as pd


def get_driver():
  # load the chrome driver with options
  chrome_driver_path = "./chromedriver_win32/chromedriver.exe"  # path to the chromedriver

  chrome_options = Options()
  user_agent = ''.join(random.choices(string.ascii_lowercase,
                                      k=20))  # random user agent name
  chrome_options.add_argument(f'user-agent={user_agent}')
  # chrome_options.add_argument("--disable-extensions")
  # chrome_options.add_argument('--load-extension=extension_3_4_4_0.crx')
  chrome_options.add_extension(
    './chromedriver_win32/istilldontcareaboutcookies-chrome-1.1.1_0.crx'
  )  # to get a crx to load: https://techpp.com/2022/08/22/how-to-download-and-save-chrome-extension-as-crx/
  chrome_options.add_argument("start-maximized")
  chrome_options.add_argument("disable-infobars")
  # chrome_options.add_argument(r"--user-data-dir=/Users/klimm/AppData/Local/Google/Chrome/User Data")
  # chrome_options.add_argument(r'--profile-directory=Default')
  # chrome_options.add_argument("--no-sandbox")
  # chrome_options.add_argument("--disable-dev-shm-usage")
  # chrome_options.add_argument("--headless")     # run the script without having a browser window open
  driver = webdriver.Chrome(
    executable_path=chrome_driver_path, chrome_options=chrome_options
  )  # creates a web driver; general variable (will not be passed to a function)
  # driver.maximize_window()

  return driver


# Clean the dataset (i.e., remove duplicates)
def remove_duplicates_in_dataset(path_dataset, path_cleaned_dataset):
  dataset = pd.read_csv(
    path_dataset,
    delimiter="\t",
    encoding="utf-8",
  )
  print("Shape of uncleaned dataset: ", dataset.shape)
  clean_dataset = dataset.drop_duplicates(keep=False)
  print("Shape of cleaned dataset: ", clean_dataset.shape)
  print(
    'Cleaned dataset (simply dropped duplicates created by potential scraping issues):\n',
    clean_dataset)
  clean_dataset.to_csv(
    path_cleaned_dataset, sep="\t", encoding="utf-8",
    index=False)  # convert the pandas dataframe to a CSV file
  return clean_dataset


def convert_csv_to_xml(path_dataset_csv, path_dataset_xml):
  dataset = pd.read_csv(
    path_dataset_csv,
    delimiter="\t",
    encoding="utf-8",
  )
  dataset.to_xml(path_dataset_xml, parser="etree", index=False)
