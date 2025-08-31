import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define product catalog
products = [
    {"sku": "JEANS-001", "name": "Slim Fit Denim Jeans", "category": "Bottoms", "base_price": 89.99},
    {"sku": "TSHIRT-001", "name": "Cotton Crew Neck T-Shirt", "category": "Tops", "base_price": 29.99},
    {"sku": "SHIRT-001", "name": "Classic Dress Shirt", "category": "Tops", "base_price": 59.99},
    {"sku": "HOODIE-001", "name": "Premium Zip Hoodie", "category": "Outerwear", "base_price": 79.99},
    {"sku": "JACKET-001", "name": "Denim Jacket", "category": "Outerwear", "base_price": 99.99},
    {"sku": "SKIRT-001", "name": "A-Line Midi Skirt", "category": "Bottoms", "base_price": 49.99},
    {"sku": "DRESS-001", "name": "Summer Maxi Dress", "category": "Dresses", "base_price": 69.99},
    {"sku": "SWEATER-001", "name": "Cable Knit Sweater", "category": "Tops", "base_price": 89.99},
    {"sku": "SHORTS-001", "name": "Chino Shorts", "category": "Bottoms", "base_price": 39.99},
    {"sku": "ACCESSORY-001", "name": "Leather Belt", "category": "Accessories", "base_price": 34.99},
]

# Define sizes and colors
sizes = ["XS", "S", "M", "L", "XL", "XXL", "One Size"]
colors = ["Black", "White", "Navy", "Gray", "Red", "Blue", "Green", "Brown", "Beige", "Pink"]

# Generate date range (2 years of data)
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date)

# Generate sales data
sales_data = []
order_id = 1000

for date in date_range:
    # Determine number of orders for this day (seasonality and weekday patterns)
    if date.month in [11, 12]:  # Holiday season
        daily_orders = random.randint(40, 60)
    elif date.month in [6, 7, 8]:  # Summer season
        daily_orders = random.randint(30, 50)
    else:
        daily_orders = random.randint(20, 40)
    
    # Fewer orders on weekends for B2B, more for B2C
    if date.weekday() >= 5:  # Weekend
        daily_orders = int(daily_orders * 0.7)
    else:
        daily_orders = int(daily_orders * 1.2)
    
    for order in range(daily_orders):
        order_id += 1
        order_time = datetime.combine(date, datetime.min.time()) + timedelta(
            hours=random.randint(9, 21),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Select 1-4 items per order
        num_items = random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0]
        
        for item in range(num_items):
            product = random.choice(products)
            sku = product["sku"]
            
            # Determine quantity (mostly 1, sometimes 2-3)
            quantity = 1
            if random.random() < 0.15:  # 15% chance of multiple items
                quantity = random.randint(2, 3)
            
            # Get size and color
            size = random.choice(sizes) if product["category"] != "Accessories" else "One Size"
            color = random.choice(colors)
            
            # Calculate price with some variation (sales, discounts, etc.)
            price_variation = random.uniform(0.7, 1.1)  # 30% discount to 10% premium
            # Seasonal pricing adjustments
            if date.month in [11, 12]:  # Holiday season premium
                price_variation *= random.uniform(1.0, 1.2)
            elif date.month in [1, 7]:  # Sale season
                price_variation *= random.uniform(0.7, 0.9)
                
            unit_price = round(product["base_price"] * price_variation, 2)
            revenue = unit_price * quantity
            
            sales_data.append({
                "order_id": order_id,
                "order_date": order_time,
                "sku": sku,
                "product_name": product["name"],
                "category": product["category"],
                "color": color,
                "size": size,
                "unit_price": unit_price,
                "quantity": quantity,
                "revenue": revenue
            })

# Create DataFrame
df = pd.DataFrame(sales_data)

# Add some missing values randomly (real-world data often has missing values)
missing_mask = np.random.random(len(df)) < 0.02  # 2% missing values
df.loc[missing_mask, 'color'] = None

missing_mask = np.random.random(len(df)) < 0.01  # 1% missing values
df.loc[missing_mask, 'size'] = None

# Save to CSV
df.to_csv("fashion_sales_dataset.csv", index=False)

print(f"Generated {len(df)} sales records")
print(f"Time period: {df['order_date'].min()} to {df['order_date'].max()}")
print(f"Unique SKUs: {df['sku'].nunique()}")
print(f"Total revenue: ${df['revenue'].sum():,.2f}")

# Show sample of data
print("\nSample of the dataset:")
print(df.head(10))