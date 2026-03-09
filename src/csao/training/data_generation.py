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
    'Pizza': {
        'main': ['Margherita Pizza', 'Pepperoni Pizza', 'Farmhouse Pizza'],
        'side': ['Garlic Breadsticks', 'Cheese Dip', 'Chicken Wings'],
        'dessert': ['Choco Lava Cake', 'Brownie'],
        'drink': ['Coke', 'Sprite', 'Iced Tea']
    }
}

COMPLEMENT_RULES = {
    'Chicken Biryani': ['Raita', 'Salan', 'Gulab Jamun', 'Lassi', 'Thumbs Up'],
    'Burger': ['Fries', 'Soft Drink', 'Milkshake', 'Onion Rings'],
    'Margherita Pizza': ['Garlic Breadsticks', 'Coke']
}

def build_catalog() -> Dict:
    item_info = {}
    for c, categories in CUISINE_MAPPING.items():
        for cat, items in categories.items():
            for item in items:
                item_info[item] = cat
    return item_info

def generate_synthetic_sessions(df_clean: pd.DataFrame, num_sessions: int = 5000):
    item_info = build_catalog()
    all_items = list(item_info.keys())
    sessions = []
    
    for i in range(num_sessions):
        # Focus on items with rules to ensure some positives
        base_item = random.choice(['Chicken Biryani', 'Burger', 'Margherita Pizza', 'Butter Chicken'])
        cart_items = [base_item] + random.sample(all_items, random.randint(0, 2))
        
        # 10 candidates per session
        for _ in range(10):
            candidate = random.choice(all_items)
            # Higher probability if it's a complement
            is_comp = candidate in COMPLEMENT_RULES.get(base_item, []) or random.random() < 0.05
            label = 1 if is_comp else 0
            
            sessions.append({
                'session_id': i,
                'restaurant_id': 1,
                'cuisine': 'North Indian',
                'area': 'Koramangala',
                'cost_tier': 1,
                'cart_items': '|'.join(cart_items),
                'candidate_item': candidate,
                'candidate_category': item_info.get(candidate, 'other'),
                'label': label,
                'meal_time': 'dinner',
                'cart_size': len(cart_items),
                'hour_of_day': 19
            })
    return pd.DataFrame(sessions)

def main():
    print("Generating benchmark datasets...")
    df = generate_synthetic_sessions(None)
    df.to_csv('test_sessions.csv', index=False)
    print(f"Saved {len(df)} rows to test_sessions.csv")

if __name__ == "__main__":
    main()
