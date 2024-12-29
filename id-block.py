from playwright.sync_api import sync_playwright
import json
import re
"""
Get the IDs and names of objects in Minecraft from the "https://minecraft-ids.grahamedgecombe.com/" page
"""
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://minecraft-ids.grahamedgecombe.com/")
    elements = page.query_selector_all(".row")
    
    ids = {}
    
    with open("results.json", "w") as f:
        print(len(elements))
        for element in elements:
            
            # Replace line breaks and excess spaces
            string = element.text_content().replace("\n", " ").strip()
            
            print(f"Cleaned string: {string}")  # For debugging
            
            # Adjusted regular expression
            pattern = r'(\d+(:\d+)?)\s+([a-zA-Z0-9_ ]+)\(([^)]+)\)'
            
            # Search for matches
            matches = re.search(pattern, string)
            if matches:
                # Find the ID (key)
                key = matches.group(1)
                # Find the value (Minecraft ID)
                value = matches.group(4)
                value = re.sub("^minecraft:", "", value)
                # Save to the dictionary
                ids[key] = value
            else:
                print("No match found")
        
        # Display the content of the dictionary
        print(ids)
        
        # Save the results in JSON format
        json.dump(ids, f, indent=4)
    
    browser.close()
