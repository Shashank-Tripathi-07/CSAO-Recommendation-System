import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import os
from datetime import datetime
import json

random.seed(42)
np.random.seed(42)

# Cuisine mapping definition
CUISINE_MAPPING = {
    'North Indian': {
        'main': ['Butter Chicken', 'Paneer Tikka Masala', 'Dal Makhani', 'Naan', 'Roti'],
        'side': ['Raita', 'Papad', 'Kachumber Salad'],
        'dessert': ['Gulab Jamun', 'Rasgulla'],
        'drink': ['Lassi', 'Jaljeera', 'Masala Chas']
    },
    'South Indian': {
        'main': ['Masala Dosa', 'Idli', 'Medu Vada', 'Uttapam'],
        'side': ['Sambar', 'Coconut Chutney', 'Tomato Chutney'],
        'dessert': ['Payasam', 'Mysore Pak'],
        'drink': ['Filter Coffee', 'Buttermilk']
    },
    'Chinese': {
        'main': ['Hakka Noodles', 'Fried Rice', 'Manchurian', 'Chilli Chicken'],
        'side': ['Spring Rolls', 'Momos', 'Chilli Potato'],
        'dessert': ['Honey Noodles with Ice Cream', 'Darsaan'],
        'drink': ['Jasmine Tea', 'Coke', 'Iced Tea']
    },
    'Fast Food': {
        'main': ['Burger', 'Veggie Burger', 'Chicken Wrap'],
        'side': ['Fries', 'Onion Rings', 'Chicken Nuggets'],
        'dessert': ['Brownie', 'Soft Serve'],
        'drink': ['Soft Drink', 'Milkshake', 'Cold Coffee']
    },
    'Biryani': {
        'main': ['Chicken Biryani', 'Mutton Biryani', 'Veg Biryani'],
        'side': ['Raita', 'Salan', 'Boiled Egg'],
        'dessert': ['Double ka Meetha', 'Gulab Jamun'],
        'drink': ['Thumbs Up', 'Lassi']
    },
    'Continental': {
        'main': ['Grilled Chicken Breast', 'Pasta Alfredo', 'Steak', 'Roast Roast vegetable'],
        'side': ['Mashed Potatoes', 'Garlic Bread', 'Sautéed Veggies'],
        'dessert': ['Cheesecake', 'Tiramisu'],
        'drink': ['Red Wine', 'Mojito', 'Lemonade']
    },
    'Desserts': {
        'main': ['Chocolate Cake', 'Red Velvet Cupcake', 'Waffles'],
        'side': ['Extra Syrups', 'Sprinkles'],
        'dessert': ['Ice Cream Scoop', 'Fruit Tart'],
        'drink': ['Hot Chocolate', 'Latte']
    },
    'Beverages': {
        'main': ['Cold Coffee', 'Frappe', 'Smoothie'],
        'side': ['Cookies', 'Muffin'],
        'dessert': ['Donut', 'Croissant'],
        'drink': ['Espresso', 'Cappuccino']
    },
    'Pizza': {
        'main': ['Margherita Pizza', 'Pepperoni Pizza', 'Farmhouse Pizza'],
        'side': ['Garlic Breadsticks', 'Cheese Dip', 'Chicken Wings'],
        'dessert': ['Choco Lava Cake', 'Brownie'],
        'drink': ['Coke', 'Sprite', 'Iced Tea']
    },
    'Rolls': {
        'main': ['Chicken Tikka Roll', 'Paneer Roll', 'Egg Roll'],
        'side': ['French Fries', 'Extra Dip'],
        'dessert': ['Phirni', 'Gulab Jamun'],
        'drink': ['Cold Drink', 'Nimbu Pani']
    }
}

# Base prices for items based on tier
PRICE_TIERS = {
    0: {'main': 150, 'side': 60, 'dessert': 80, 'drink': 50},  # Budget
    1: {'main': 300, 'side': 120, 'dessert': 150, 'drink': 100}, # Mid
    2: {'main': 600, 'side': 250, 'dessert': 300, 'drink': 200}  # Premium
}

COMPLEMENT_RULES = {
    'Chicken Biryani': ['Raita', 'Salan', 'Gulab Jamun', 'Lassi', 'Thumbs Up'],
    'Mutton Biryani': ['Raita', 'Salan', 'Double ka Meetha', 'Thumbs Up'],
    'Veg Biryani': ['Raita', 'Salan', 'Gulab Jamun'],
    'Burger': ['Fries', 'Soft Drink', 'Milkshake', 'Onion Rings'],
    'Veggie Burger': ['Fries', 'Soft Drink', 'Milkshake'],
    'Margherita Pizza': ['Garlic Breadsticks', 'Cheese Dip', 'Coke', 'Choco Lava Cake'],
    'Pepperoni Pizza': ['Garlic Breadsticks', 'Chicken Wings', 'Coke'],
    'Masala Dosa': ['Sambar', 'Coconut Chutney', 'Filter Coffee'],
    'Idli': ['Sambar', 'Coconut Chutney', 'Filter Coffee', 'Medu Vada'],
    'Butter Chicken': ['Naan', 'Roti', 'Papad', 'Lassi'],
    'Paneer Tikka Masala': ['Naan', 'Roti', 'Raita', 'Dal Makhani'],
    'Hakka Noodles': ['Manchurian', 'Chilli Chicken', 'Momos', 'Coke'],
    'Fried Rice': ['Manchurian', 'Chilli Chicken', 'Spring Rolls'],
    'Grilled Chicken Breast': ['Mashed Potatoes', 'Sautéed Veggies', 'Red Wine'],
    'Pasta Alfredo': ['Garlic Bread', 'Mojito', 'Tiramisu'],
    'Chicken Tikka Roll': ['French Fries', 'Cold Drink'],
    'Paneer Roll': ['French Fries', 'Cold Drink']
}


def clean_zomato_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing cuisine or location
    df = df.dropna(subset=['cuisines', 'location'])
    
    # Parse cuisines into list
    df['cuisines_list'] = df['cuisines'].apply(lambda x: [c.strip() for c in str(x).split(',')])
    
    # Normalize rating
    def parse_rating(r):
        try:
            r = str(r).strip()
            if r == 'NEW' or r == '-' or r == '':
                return np.nan
            return float(r.split('/')[0])
        except:
            return np.nan
    df['rating'] = df['rate'].apply(parse_rating)
    # Fill missing ratings with mean for synthetic data purposes
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    
    # Bin cost
    def parse_cost(c):
        try:
            return float(str(c).replace(',', ''))
        except:
            return np.nan
    df['cost'] = df['approx_cost(for two people)'].apply(parse_cost)
    df['cost'] = df['cost'].fillna(df['cost'].mean())
    df['cost_tier'] = pd.cut(df['cost'], bins=[0, 400, 800, np.inf], labels=[0, 1, 2], right=False).astype(int)
    
    # Create unique restaurant ID
    df['restaurant_id'] = range(1, len(df) + 1)
    
    # Filter to only cuisines we have in mapping
    valid_cuisines = set(CUISINE_MAPPING.keys())
    
    # Explode to get one row per restaurant per valid cuisine
    df_exploded = df.explode('cuisines_list')
    df_clean = df_exploded[df_exploded['cuisines_list'].isin(valid_cuisines)].copy()
    
    df_clean.rename(columns={'cuisines_list': 'cuisine', 'location': 'area'}, inplace=True)
    return df_clean[['restaurant_id', 'cuisine', 'area', 'cost_tier', 'rating']]


def build_catalog() -> Dict:
    # Build complete item-to-category and item-to-tier mappings for quick lookups
    item_info = {}
    for c, categories in CUISINE_MAPPING.items():
        for cat, items in categories.items():
            for item in items:
                item_info[item] = cat
    return item_info

def get_meal_time(hour: int) -> str:
    if 6 <= hour < 11:
        return 'breakfast'
    elif 11 <= hour < 16:
        return 'lunch'
    elif 16 <= hour < 22:
        return 'dinner'
    else:
        return 'late_night'

def calculate_cart_price_tier(cart_list: List[str], item_info: Dict) -> int:
    return 1 # simplify for simulation; cost tier is at restaurant level, we'll use restaurant cost tier for calculation 


def generate_synthetic_sessions(df_clean: pd.DataFrame, num_sessions: int = 100000) -> pd.DataFrame:
    item_info = build_catalog()
    
    restaurants = df_clean.to_dict('records')
    sessions = []
    
    start_time_base = datetime(2023, 1, 1).timestamp()
    time_increment = (365 * 24 * 3600) / num_sessions # Spread over a year
    
    for i in tqdm(range(num_sessions), desc="Generating Sessions"):
        rest = random.choice(restaurants)
        cuisine = rest['cuisine']
        menu = CUISINE_MAPPING[cuisine]
        
        all_items = []
        for cat, items in menu.items():
            all_items.extend(items)
            
        if not all_items:
            continue
            
        cart_size = random.randint(1, min(3, len(all_items)))
        cart_items = random.sample(all_items, cart_size)
        
        candidates = list(set(all_items) - set(cart_items))
        
        hour = random.randint(0, 23)
        meal_time = get_meal_time(hour)
        
        # Calculate acceptance probability for candidates
        for candidate in candidates:
            prob = 0.15 # base
            
            # +0.35 if complement to ANY cart item
            is_complement = False
            for c_item in cart_items:
                if candidate in COMPLEMENT_RULES.get(c_item, []):
                    is_complement = True
                    break
            
            if is_complement:
                prob += 0.35
                
            # +0.10 if cart size == 1
            if cart_size == 1:
                prob += 0.10
                
            # +0.05 if restaurant rating > 4.0
            if rest['rating'] > 4.0:
                prob += 0.05
                
            # -0.05 if candidate price tier > cart average (simplified: use 0 since we apply price based on rest tier, 
            # let's assume candidate tier variation depends on category. Mains > Sides > Drinks/Desserts.
            cat_rank = {'main': 3, 'side': 2, 'dessert': 2, 'drink': 1}
            cart_avg_rank = sum(cat_rank[item_info[x]] for x in cart_items) / cart_size
            candidate_rank = cat_rank[item_info[candidate]]
            if candidate_rank > cart_avg_rank:
                prob -= 0.05
                
            prob = min(prob, 0.85)
            prob = max(prob, 0.0) # Ensure no negative
            
            label = 1 if random.random() < prob else 0
            
            sessions.append({
                'session_id': i,
                'timestamp': start_time_base + (i * time_increment),
                'restaurant_id': rest['restaurant_id'],
                'cuisine': cuisine,
                'area': rest['area'],
                'cost_tier': rest['cost_tier'],
                'rating': rest['rating'],
                'cart_items': '|'.join(cart_items),
                'candidate_item': candidate,
                'candidate_category': item_info[candidate],
                'cart_size': cart_size,
                'hour_of_day': hour,
                'meal_time': meal_time,
                'label': label,
                'is_complement': is_complement, # saving this for easier feature engineering later, though technically could recompute
            })
            
    return pd.DataFrame(sessions)

def main():
    zomato_path = '../zomato.csv'  # Adjust path as needed
    if not os.path.exists(zomato_path):
        print(f"Warning: {zomato_path} not found. Please place the Kaggle Zomato dataset there.")
        # Create a mock tiny dataset if it doesn't exist just to test the pipeline
        mock_data = pd.DataFrame({
            'cuisines': ['North Indian', 'South Indian', 'Chinese', 'Fast Food, Burger'],
            'location': ['Banashankari', 'Indiranagar', 'Koramangala', 'Whitefield'],
            'rate': ['4.1/5', '3.8/5', '4.5/5', 'NEW'],
            'approx_cost(for two people)': ['800', '300', '1,200', '500']
        })
        mock_data.to_csv('mock_zomato.csv', index=False)
        zomato_path = 'mock_zomato.csv'
        print("Using mock_zomato.csv for demonstration.")

    print("Cleaning dataset...")
    df_clean = clean_zomato_data(zomato_path)
    
    print(f"Generating 100,000 synthetic sessions from {len(df_clean)} valid restaurants...")
    df_sessions = generate_synthetic_sessions(df_clean, num_sessions=100000)
    
    out_path = 'synthetic_sessions.csv'
    df_sessions.to_csv(out_path, index=False)
    print(f"Saved {len(df_sessions)} candidates to {out_path}.")
    
    # Export catalogs for usage in features/inference
    item_info = build_catalog()
    with open('item_catalog.json', 'w') as f:
        json.dump(item_info, f)

if __name__ == "__main__":
    main()
