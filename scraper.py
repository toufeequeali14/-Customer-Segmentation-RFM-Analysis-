# scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Since we are scraping a local file, we don't use requests.get()
# We open the file and read its content
file_path = 'F:/BecomeDataScientist/Datasets/webScrapping/Real.html'

# Check if the file exists before proceeding
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found. Please create it first.")
    exit()

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Create a BeautifulSoup object to parse the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Find all the job listing containers on the page
# This is the most important step: we find the common parent element for each job.
job_elements = soup.find_all('div', class_='job-listing')

# Initialize an empty list to store our extracted job data
jobs_data = []

# Loop through each job element and extract the details
for job in job_elements:
    # .find() gets the first matching element. .get_text() extracts the text.
    # Using .strip() to clean up any extra whitespace
    title = job.find('h2', class_='job-title').get_text(strip=True)
    company = job.find('p', class_='company').get_text(strip=True).replace('Company:', '')
    location = job.find('p', class_='location').get_text(strip=True).replace('Location:', '')
    date_posted = job.find('p', class_='date-posted').get_text(strip=True).replace('Posted:', '')
    description = job.find('div', class_='job-description').get_text(strip=True)

    # Append the extracted data as a dictionary to our list
    jobs_data.append({
        'Title': title,
        'Company': company,
        'Location': location,
        'Date Posted': date_posted,
        'Description': description
    })

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(jobs_data)

# Print the DataFrame to the console to verify it worked
print("Scraped Job Data:")
print(df)

# Save the DataFrame to a CSV file for the next step (analysis)
df.to_csv(r'F:\BecomeDataScientist\Datasets\webScrapping\scraped_jobs.csv', index=False)
print("\nData successfully saved to 'scraped_jobs.csv'")