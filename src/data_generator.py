import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def generate_data(n_samples=5000):
    np.random.seed(42)
    
    categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports']
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    product_category = np.random.choice(categories, n_samples)
    original_price = np.random.uniform(10, 1000, n_samples)
    proposed_discount = np.random.uniform(0, 0.7, n_samples)
    historical_sales = np.random.uniform(50, 5000, n_samples)
    customer_demand = np.random.uniform(0.1, 1.0, n_samples)
    season = np.random.choice(seasons, n_samples)
    promotion_status = np.random.choice([0, 1], n_samples)
    competitor_price = original_price * np.random.uniform(0.8, 1.2, n_samples)
    
    base_sales = historical_sales * customer_demand
    discount_effect = 1 + (proposed_discount * 2.5) - (proposed_discount ** 2)
    competitor_effect = (competitor_price / original_price)
    promotion_effect = 1 + (promotion_status * 0.3)
    seasonal_effect = np.where(season == 'Winter', 1.2, 
                      np.where(season == 'Summer', 1.1, 1.0))
    cat_effect_dict = {'Electronics': 1.1, 'Clothing': 1.0, 'Home': 0.9, 'Beauty': 1.05, 'Sports': 0.95}
    category_effect = np.array([cat_effect_dict[c] for c in product_category])
    
    expected_sales = (base_sales * discount_effect * competitor_effect * 
                      promotion_effect * seasonal_effect * category_effect)
    
    noise = np.random.normal(0, expected_sales * 0.1, n_samples)
    expected_sales = np.maximum(0, expected_sales + noise)
    
    df = pd.DataFrame({
        'Product_Category': product_category,
        'Original_Price': original_price,
        'Proposed_Discount': proposed_discount,
        'Historical_Sales': historical_sales,
        'Customer_Demand': customer_demand,
        'Season': season,
        'Promotion_Status': promotion_status,
        'Competitor_Price': competitor_price,
        'Expected_Sales': expected_sales
    })
    
    output_path = os.path.join(DATA_DIR, 'retail_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_data()
