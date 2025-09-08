from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np
import time
import random
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.linear_model import LinearRegression

def scrape_indeed_selenium(job_title, location, max_pages=35):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    all_jobs = []
    failed_pages = 0
    max_failures = 5
    
    try:
        for page in range(0, max_pages * 10, 10):
            if failed_pages >= max_failures:
                print(f"Stopping after {max_failures} consecutive failed pages")
                break
                
            url = f"https://www.indeed.com/jobs?q={job_title.replace(' ', '+')}&l={location.replace(' ', '+').replace(',', '%2C')}&start={page}"
            print(f"Scraping page {(page//10) + 1}/{max_pages}: {url}")
            
            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".job_seen_beacon, .cardOutline, [data-jk]"))
                )
                time.sleep(random.uniform(2, 4))
                
                job_cards = driver.find_elements(By.CSS_SELECTOR, ".job_seen_beacon, .cardOutline, [data-jk]")
                
                if not job_cards:
                    print(f"No job cards found on page {(page//10) + 1}.")
                    failed_pages += 1
                    continue
                
                print(f"Found {len(job_cards)} job cards on page {(page//10) + 1}")
                failed_pages = 0
                
                page_jobs = 0
                for job in job_cards:
                    try:
                        title = extract_text(job, [".jobTitle", "h2", "a.jcs-JobTitle", "[data-testid='jobTitle']"])
                        company = extract_text(job, [".companyName", ".company", "[data-testid='company-name']"])
                        location_text = extract_text(job, [".companyLocation", ".location", "[data-testid='text-location']"])
                        
                        # EXTRACT DATE POSTED - CRITICAL FOR TIME SERIES
                        date_text = extract_text(job, [".date", ".datePosted", ".result-link-bar", ".posted-since"])
                        posted_date = parse_date(date_text)
                        
                        salary = extract_text(job, [".salary-snippet", ".salaryText", ".attribute-snippet"])
                        snippet = extract_text(job, [".job-snippet", ".summary", ".job-snippet__list"])
                        
                        job_url = extract_attribute(job, "href", [".jcs-JobTitle", "a", "[data-jk]"])
                        if job_url and not job_url.startswith("http"):
                            job_url = "https://www.indeed.com" + job_url
                        
                        job_id = extract_attribute(job, "data-jk", ["[data-jk]"]) or f"{title}_{company}_{posted_date}"
                        
                        if title and title != "N/A":
                            all_jobs.append({
                                'Job_ID': job_id,
                                'Title': title,
                                'Company': company,
                                'Location': location_text,
                                'Date_Posted_Text': date_text,
                                'Date_Posted': posted_date,
                                'Salary': salary,
                                'Snippet': snippet[:150] + "..." if snippet and len(snippet) > 150 else snippet,
                                'URL': job_url,
                                'Scrape_Date': datetime.now().strftime("%Y-%m-%d"),
                                'Scrape_Timestamp': datetime.now(),
                                'Page': page//10 + 1
                            })
                            page_jobs += 1
                            
                    except Exception as e:
                        continue
                
                print(f"Successfully extracted {page_jobs} jobs from page {(page//10) + 1}")
                time.sleep(random.uniform(3, 6))
                
            except TimeoutException:
                print(f"Timeout on page {(page//10) + 1}")
                failed_pages += 1
                continue
            except Exception as e:
                print(f"Error loading page {(page//10) + 1}: {e}")
                failed_pages += 1
                continue
                
    except Exception as e:
        print(f"Fatal error: {e}")
    
    finally:
        driver.quit()
    
    return all_jobs

def parse_date(date_text):
    """Parse relative dates like 'Posted today', '30+ days ago', etc."""
    if not date_text or date_text == "N/A":
        return None
    
    date_text = date_text.lower()
    today = datetime.now()
    
    # Remove "Posted" prefix if present
    date_text = re.sub(r'^posted\s*', '', date_text)
    
    if 'today' in date_text or 'just now' in date_text:
        return today.strftime("%Y-%m-%d")
    elif 'yesterday' in date_text:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif 'days ago' in date_text:
        numbers = re.findall(r'\d+', date_text)
        if numbers:
            days_ago = int(numbers[0])
            return (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    elif 'month' in date_text or 'months ago' in date_text:
        numbers = re.findall(r'\d+', date_text)
        if numbers:
            months_ago = int(numbers[0])
            return (today - timedelta(days=30*months_ago)).strftime("%Y-%m-%d")
    elif 'week' in date_text or 'weeks ago' in date_text:
        numbers = re.findall(r'\d+', date_text)
        if numbers:
            weeks_ago = int(numbers[0])
            return (today - timedelta(days=7*weeks_ago)).strftime("%Y-%m-%d")
    
    return None

def extract_text(element, selectors):
    for selector in selectors:
        try:
            elem = element.find_element(By.CSS_SELECTOR, selector)
            text = elem.text.strip()
            if text:
                return text
        except NoSuchElementException:
            continue
    return "N/A"

def extract_attribute(element, attribute, selectors):
    for selector in selectors:
        try:
            elem = element.find_element(By.CSS_SELECTOR, selector)
            attr_value = elem.get_attribute(attribute)
            if attr_value:
                return attr_value
        except NoSuchElementException:
            continue
    return "N/A"

def remove_duplicates(jobs):
    seen = set()
    unique_jobs = []
    for job in jobs:
        if job['Job_ID'] not in seen:
            seen.add(job['Job_ID'])
            unique_jobs.append(job)
    return unique_jobs

# ================= TIME SERIES ANALYSIS FUNCTIONS =================

class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_time_series_data()
    
    def prepare_time_series_data(self):
        """Prepare data for time series analysis"""
        # Convert to datetime and filter valid dates
        self.df['Date_Posted'] = pd.to_datetime(self.df['Date_Posted'], errors='coerce')
        self.df = self.df.dropna(subset=['Date_Posted'])
        
        # Create daily time series
        self.daily_series = self.df.groupby('Date_Posted').size()
        self.daily_series = self.daily_series.asfreq('D', fill_value=0)
        
        # Create weekly and monthly aggregates
        self.weekly_series = self.daily_series.resample('W').sum()
        self.monthly_series = self.daily_series.resample('M').sum()
    
    def check_stationarity(self, series):
        """Check if time series is stationary using Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    
    def decompose_series(self, series, period=7):
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            decomposition = seasonal_decompose(series.dropna(), period=min(period, len(series)//2))
            return decomposition
        except:
            return None
    
    def detect_trend(self):
        """Detect overall trend using linear regression"""
        dates = np.arange(len(self.daily_series))
        values = self.daily_series.values
        
        valid_mask = ~np.isnan(values)
        X = dates[valid_mask].reshape(-1, 1)
        y = values[valid_mask]
        
        if len(X) < 2:
            return None, None
        
        model = LinearRegression()
        model.fit(X, y)
        trend_slope = model.coef_[0]
        r_squared = model.score(X, y)
        
        return trend_slope, r_squared
    
    def detect_seasonality(self, period=7):
        """Detect weekly seasonality"""
        if len(self.daily_series) < 2 * period:
            return None, None
        
        day_of_week = self.daily_series.index.dayofweek
        weekly_pattern = self.daily_series.groupby(day_of_week).mean()
        
        # Statistical significance test
        groups = [self.daily_series[day_of_week == i] for i in range(7)]
        if all(len(group) > 1 for group in groups):
            f_stat, p_value = stats.f_oneway(*groups)
            return weekly_pattern, p_value
        
        return weekly_pattern, None

def visualize_time_series(analyzer):
    """Create comprehensive time series visualizations"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Daily Trend Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Raw daily data
    analyzer.daily_series.plot(ax=ax1, title='Daily Job Postings', linewidth=1, color='blue')
    ax1.set_ylabel('Number of Jobs')
    ax1.grid(True, alpha=0.3)
    
    # 7-day moving average
    rolling_mean = analyzer.daily_series.rolling(window=7).mean()
    rolling_mean.plot(ax=ax2, title='7-Day Moving Average', linewidth=2, color='red')
    ax2.set_ylabel('Moving Average')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative trend
    cumulative = analyzer.daily_series.cumsum()
    cumulative.plot(ax=ax3, title='Cumulative Job Postings', linewidth=2, color='green')
    ax3.set_ylabel('Cumulative Count')
    ax3.grid(True, alpha=0.3)
    
    # Weekly pattern (if available)
    weekly_pattern, _ = analyzer.detect_seasonality()
    if weekly_pattern is not None:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_pattern.index = range(7)
        weekly_pattern.plot(kind='bar', ax=ax4, title='Average Posts by Day of Week', color='orange')
        ax4.set_xticklabels(days)
        ax4.set_ylabel('Average Posts')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('job_market_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Seasonal Decomposition
    decomposition = analyzer.decompose_series(analyzer.daily_series)
    if decomposition:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=ax1, title='Observed')
        decomposition.trend.plot(ax=ax2, title='Trend')
        decomposition.seasonal.plot(ax=ax3, title='Seasonal')
        decomposition.resid.plot(ax=ax4, title='Residual')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_analysis_report(analyzer):
    """Generate comprehensive analysis report"""
    print("\n" + "="*60)
    print("JOB MARKET TIME SERIES ANALYSIS REPORT")
    print("="*60)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total jobs analyzed: {len(analyzer.df)}")
    print(f"Time period: {analyzer.daily_series.index.min().strftime('%Y-%m-%d')} to {analyzer.daily_series.index.max().strftime('%Y-%m-%d')}")
    print(f"Average daily posts: {analyzer.daily_series.mean():.2f}")
    print(f"Maximum daily posts: {analyzer.daily_series.max()}")
    print(f"Minimum daily posts: {analyzer.daily_series.min()}")
    
    # Trend analysis
    trend_slope, trend_strength = analyzer.detect_trend()
    print(f"\n📈 TREND ANALYSIS:")
    print(f"Trend direction: {'↗️ Increasing' if trend_slope > 0 else '↘️ Decreasing'}")
    print(f"Trend slope: {trend_slope:.3f} jobs per day")
    print(f"Trend strength (R²): {trend_strength:.3%}")
    
    # Stationarity test
    stationarity = analyzer.check_stationarity(analyzer.daily_series)
    print(f"\n📉 STATIONARITY TEST:")
    print(f"ADF Statistic: {stationarity['adf_statistic']:.3f}")
    print(f"p-value: {stationarity['p_value']:.4f}")
    print(f"Series is stationary: {'✅ Yes' if stationarity['is_stationary'] else '❌ No'}")
    
    # Seasonality analysis
    weekly_pattern, p_value = analyzer.detect_seasonality()
    if weekly_pattern is not None:
        print(f"\n📅 SEASONALITY ANALYSIS:")
        print(f"Weekly pattern p-value: {p_value:.4f}")
        print(f"Weekly pattern significant: {'✅ Yes' if p_value and p_value < 0.05 else '❌ No'}")
        print("\nAverage posts by day:")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, count in enumerate(weekly_pattern):
            print(f"  {days[i]}: {count:.2f} posts")
    
    print(f"\n💡 INSIGHTS:")
    if trend_slope > 0.1:
        print("• Strong positive trend in job postings 📈")
    elif trend_slope < -0.1:
        print("• Declining trend in job postings 📉")
    else:
        print("• Stable job market trend ↔️")
    
    if weekly_pattern is not None and p_value and p_value < 0.05:
        best_day = weekly_pattern.idxmax()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(f"• Best day for job hunting: {days[best_day]} 🎯")

# Main execution
if __name__ == "__main__":
    # Search parameters
    job_title = "data scientist"
    location = "New York, NY"
    target_jobs = 500
    pages_needed = target_jobs // 15 + 5
    
    print(f"Scraping Indeed for '{job_title}' in '{location}'...")
    print(f"Target: {target_jobs} jobs (~{pages_needed} pages)")
    
    # Scrape jobs
    jobs_data = scrape_indeed_selenium(job_title, location, max_pages=pages_needed)
    
    # Remove duplicates
    unique_jobs = remove_duplicates(jobs_data)
    
    # Create DataFrame
    if unique_jobs:
        df = pd.DataFrame(unique_jobs)
        
        print(f"\n✅ Successfully scraped {len(df)} unique jobs:")
        print(df[['Title', 'Company', 'Date_Posted']].head(10).to_string(index=False))
        
        # Save to CSV
        csv_path = r'F:\BecomeDataScientist\Datasets\indeed_web_scrapping\indeed_jobs_with_dates.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n💾 Data saved to '{csv_path}'")
        
        # ================= TIME SERIES ANALYSIS =================
        print(f"\n{"="*50}")
        print("PERFORMING TIME SERIES ANALYSIS...")
        print("="*50)
        
        # Initialize analyzer
        analyzer = TimeSeriesAnalyzer(df)
        
        # Generate visualizations
        visualize_time_series(analyzer)
        
        # Generate comprehensive report
        generate_analysis_report(analyzer)
        
        # Show basic statistics
        print(f"\n📋 DATASET STATISTICS:")
        print(f"Total unique jobs: {len(df)}")
        print(f"Unique companies: {df['Company'].nunique()}")
        print(f"Unique locations: {df['Location'].nunique()}")
        print(f"Date range: {df['Date_Posted'].min()} to {df['Date_Posted'].max()}")
        
    else:
        print("❌ No jobs were scraped.")