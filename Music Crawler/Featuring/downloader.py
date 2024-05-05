import os
import re 

import IPython
from IPython.display import display
from IPython.display import display, HTML

import spider


PLAYLIST_LINK = "user input"
GENRE = 'user input'

CWD = os.getcwd()
LOCATION = os.path.join(CWD,'MUSIC')
if os.path.isdir(LOCATION)==False:
    os.mkdir(LOCATION)

def returnSPOT_ID(link):
        # # The 'returnSPOT_ID' function from your scraper code

        # Define the regular expression pattern for the Spotify playlist URL
        pattern = r"https://open\.spotify\.com/playlist/([a-zA-Z0-9]+)\?si=.*"

        # Try to match the pattern in the input text
        match = re.match(pattern, link)

        if not match:
            return False
        return True

def handle_link_button_click(playlist_link):
    global SPOTIFY_PLAYLIST_LINK
    # Get the link from the input field
    if returnSPOT_ID(playlist_link):
        SPOTIFY_PLAYLIST_LINK = playlist_link
        # Display the entered link as a clickable HTML link
        display(HTML(f"Playlist Link Entered : <a href='{SPOTIFY_PLAYLIST_LINK}' target='_blank'>{SPOTIFY_PLAYLIST_LINK}</a>"))
        # Store the entered link as a global variable
        IPython.get_ipython().run_line_magic("store", "SPOTIFY_PLAYLIST_LINK")
    else:
        print('[*] Something Not Right about that link...  Try Again Please..')

handle_link_button_click(PLAYLIST_LINK)

if SPOTIFY_PLAYLIST_LINK is not None:
    OFFSET_VARIABLE = 0 #<-- Change to start from x number of songs
    music_folder = os.path.join(os.getcwd(), "MUSIC/"+GENRE)  # Change this path to your desired music folder (may be genres)

    scraper = spider.MusicScraper()
    ID = scraper.returnSPOT_ID(SPOTIFY_PLAYLIST_LINK)
    PLAYLIST_PATH = scraper.scrape_playlist(SPOTIFY_PLAYLIST_LINK, music_folder)
else:
    print("[*] ERROR OCCURRED. MISSING PLAYLIST LINK !")