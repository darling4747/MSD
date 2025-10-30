import webbrowser
import time

def get_location_from_browser():
    """
    Opens a webpage that uses the browser's Geolocation API to get
    the user's precise coordinates and displays them.
    """
    url = "https://www.google.com/maps/@?api=1&map_action=map&center=&zoom=15"
    print("Opening a webpage to get your location. Please allow access if prompted.")
    print("After the map loads, the URL will contain your latitude and longitude.")
    print("Look for the 'center' parameter in the URL. It will look like this:")
    print("...&center=12.9716,77.5946&...")
    print("Copy the two numbers (latitude and longitude) and paste them into your main script.")
    
    time.sleep(3) # Give the user a moment to read the instructions
    webbrowser.open(url)

if __name__ == "__main__":
    get_location_from_browser()
