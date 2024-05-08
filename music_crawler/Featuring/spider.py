import requests
import os
import string
import re
from mutagen.id3 import APIC, ID3
from mutagen.easyid3 import EasyID3


class MusicScraper():

    def __init__(self):
        super(MusicScraper, self).__init__()
        self.counter = 0  # Initialize the counter to zero
        self.session = requests.Session()

    def get_ID(self, yt_id):
        # The 'get_ID' function from your scraper code
        LINK = f'https://api.spotifydown.com/getId/{yt_id}'
        headers = {
            'authority': 'api.spotifydown.com',
            'method': 'GET',
            'path': f'/getId/{id}',
            'origin': 'https://spotifydown.com',
            'referer': 'https://spotifydown.com/',
            'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            'sec-fetch-mode': 'cors',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        response = self.session.get(url=LINK, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['id']
        return None

    def generate_Analyze_id(self, yt_id):
        # The 'generate_Analyze_id' function from your scraper code
        DL = 'https://corsproxy.io/?https://www.y2mate.com/mates/analyzeV2/ajax'
        data = {
            'k_query': f'https://www.youtube.com/watch?v={yt_id}',
            'k_page': 'home',
            'hl': 'en',
            'q_auto': 0,
        }
        headers = {
            'authority': 'corsproxy.io',
            'method': 'POST',
            'path': '/?https://www.y2mate.com/mates/analyzeV2/ajax',
            'origin': 'https://spotifydown.com',
            'referer': 'https://spotifydown.com/',
            'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            'sec-fetch-mode': 'cors',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        RES = self.session.post(url=DL, data=data, headers=headers)
        if RES.status_code == 200:
            return RES.json()
        return None

    def generate_Conversion_id(self, analyze_yt_id, analyze_id):
        # The 'generate_Conversion_id' function from your scraper code
        DL = 'https://corsproxy.io/?https://www.y2mate.com/mates/convertV2/index'
        data = {
            'vid'   : analyze_yt_id,
            'k'     : analyze_id,
        }
        headers = {
            'authority': 'corsproxy.io',
            'method': 'POST',
            'path': '/?https://www.y2mate.com/mates/analyzeV2/ajax',
            'origin': 'https://spotifydown.com',
            'referer': 'https://spotifydown.com/',
            'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            'sec-fetch-mode': 'cors',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        RES = self.session.post(url=DL, data=data, headers=headers)
        if RES.status_code == 200:
            return RES.json()
        return None

    def get_PlaylistMetadata(self, Playlist_ID):
        # The 'get_PlaylistMetadata' function from your scraper code
        URL = f"https://api.spotifydown.com/metadata/playlist/{Playlist_ID}"
        headers = {
            'authority': 'api.spotifydown.com',
            'method': 'GET',
            'path': f'/metadata/playlist/{Playlist_ID}',
            'scheme': 'https',
            'origin': 'https://spotifydown.com',
            'referer': 'https://spotifydown.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        meta_data = self.session.get(headers=headers, url=URL)
        if meta_data.status_code == 200:
            return meta_data.json()['title'] + ' - ' + meta_data.json()['artists']
        return None

    def errorcatch(self, SONG_ID):
        # The 'errorcatch' function from your scraper code
        print('[*] Trying to download...')
        headers = {
            'authority': 'api.spotifydown.com',
            'method': 'GET',
            'path': f'/download/{SONG_ID}',
            'scheme': 'https',
            'origin': 'https://spotifydown.com',
            'referer': 'https://spotifydown.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        x = self.session.get(headers=headers, url='https://api.spotifydown.com/download/' + SONG_ID)
        if x.status_code == 200:
            return x.json()['link']
        return None

    def V2catch(self, SONG_ID):
        ## Updated .. .19TH OCTOBER 2023
        # yt_id = self.get_ID(SONG_ID)

        # domain = ["co.wuk.sh", "cobalt2.snapredd.app"]
        # target_domain = domain[random.randint(0,len(domain) - 1)]
        headers = {
            "authority": "api.spotifydown.com",
            "method": "POST",
            "path": '/download/68GdZAAowWDac3SkdNWOwo',
            "scheme": "https",
            "Accept": "*/*",

            'Sec-Ch-Ua':'"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
            "Dnt": '1',
            "Origin": "https://spotifydown.com",
            "Referer": "https://spotifydown.com/",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
        }

        ## Updated .. .29TH OCTOBER 2023
        x = self.session.get(url = f'https://api.spotifydown.com/download/{SONG_ID}', headers=headers)

        # if x.status_code == 200:

        #     # par = {
        #     #     'aFormat':'"mp3"',
        #     #     'dubLang':'false',
        #     #     'filenamePattern':'"classic"',
        #     #     'isAudioOnly':'true',
        #     #     'isNoTTWatermark':'true',
        #     #     'url':f'"https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D{yt_id}"'
        #     # }

        #     file_status = self.session.post(url=f"https://{target_domain}/api/json", json=par, headers=headers)
        # print('[*] Data Gathered : ', str(x.content))
        if x.status_code == 200:

            try:
                return {
                    'link' : x.json()['link'],
                    'metadata' : None
                }
            except:
                return {
                    'link' : None,
                    'metadata' : None
                }

        return None


    def scrape_playlist(self, spotify_playlist_link, music_folder):
        ID = self.returnSPOT_ID(spotify_playlist_link)
        PlaylistName = self.get_PlaylistMetadata(ID)
        print('Playlist Name : ', PlaylistName)
        # Create Folder for Playlist
        if not os.path.exists(music_folder):
            os.makedirs(music_folder)
        try:
            FolderPath = ''.join(e for e in PlaylistName)
            playlist_folder_path = os.path.join(music_folder, FolderPath)
        except:
            playlist_folder_path = music_folder

        if not os.path.exists(playlist_folder_path):
            os.makedirs(playlist_folder_path)

        headers = {
            'authority': 'api.spotifydown.com',
            'method': 'GET',
            'path': f'/trackList/playlist/{ID}',
            'scheme': 'https',
            'accept': '*/*',
            'dnt': '1',
            'origin': 'https://spotifydown.com',
            'referer': 'https://spotifydown.com/',
            'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        }

        Playlist_Link = f'https://api.spotifydown.com/trackList/playlist/{ID}'
        offset_data = {}
        offset = 0
        offset_data['offset'] = offset

        while offset is not None:
            response = self.session.get(url=Playlist_Link, params=offset_data, headers=headers)
            if response.status_code == 200:
                Tdata = response.json()['trackList']
                page = response.json()['nextOffset']
                print("*"*100)
                c = 0
                for count, song in enumerate(Tdata):
                    c += 1
                    print(f"[{c}] Downloading : ", song['title'], "-", song['artists'])
                    filename = song['title'].translate(str.maketrans('', '', string.punctuation)) + ' - ' + song['artists'].translate(str.maketrans('', '', string.punctuation)) + '.mp3'
                    filepath = os.path.join(playlist_folder_path, filename)
                    try:
                        try:
                            V2METHOD    = self.V2catch(song['id'])
                            DL_LINK     = V2METHOD['link']
                            SONG_META   = song
                            SONG_META['file'] = filepath

                        except IndentationError:
                            yt_id = self.get_ID(song['id'])

                            if yt_id is not None:
                                data = self.generate_Analyze_id(yt_id['id'])
                                try:
                                    DL_ID = data['links']['mp3']['mp3128']['k']
                                    DL_DATA = self.generate_Conversion_id(data['vid'], DL_ID)
                                    DL_LINK = DL_DATA['dlink']
                                except Exception as NoLinkError:
                                    CatchMe = self.errorcatch(song['id'])
                                    if CatchMe is not None:
                                        DL_LINK = CatchMe
                            else:
                                print('[*] No data found for : ', song)
                        except TypeError:
                          continue

                        download_complete = False
                        try:
                          if DL_LINK is not None:
                              ## DOWNLOAD
                              link = self.session.get(DL_LINK, stream=True)
                              total_size = int(link.headers.get('content-length', 0))
                              block_size = 1024  # 1 Kilobyte
                              downloaded = 0
                              ## Save
                              with open(filepath, "wb") as f:
                                  for data in link.iter_content(block_size):
                                      f.write(data)
                                      downloaded += len(data)
                              download_complete = True
                              #Increment the counter
                              self.increment_counter()
                          else:
                              print('[*] No Download Link Found. Skipping...')
                          if (DL_LINK is not None)&(download_complete == True):
                              songTag = WritingMetaTags(tags=SONG_META, filename=filepath)
                              song_meta_add = songTag.WritingMetaTags()
                        except ConnectionError:
                            continue
                    except IndentationError as error_status:
                        print('[*] Error Status Code : ', error_status)

            if page is not None:
                offset_data['offset'] = page
                response = self.session.get(url=Playlist_Link, params=offset_data, headers=headers)
            else:
                print("*"*100)
                print('[*] Download Complete!')
                print("*"*100)
                break
        return playlist_folder_path


    def returnSPOT_ID(self, link):
        # # The 'returnSPOT_ID' function from your scraper code

        # Define the regular expression pattern for the Spotify playlist URL
        pattern = r"https://open\.spotify\.com/playlist/([a-zA-Z0-9]+)\?si=.*"

        # Try to match the pattern in the input text
        match = re.match(pattern, link)

        if not match:
            raise ValueError("Invalid Spotify playlist URL.")
        # Extract the playlist ID from the matched pattern
        extracted_id = match.group(1)

        return extracted_id

    def increment_counter(self):
            self.counter += 1

# Scraper Thread
class WritingMetaTags():
    def __init__(self, tags, filename):
        super().__init__()
        self.tags = tags
        self.filename = filename
        self.PICTUREDATA = None
        self.url = None

    def setPIC(self):
        if self.tags['cover'] is None:
            pass
        else:
            try:
                response = requests.get(self.tags['cover']+"?size=1", stream=True)
                if response.status_code == 200 :
                    audio = ID3(self.filename)
                    audio['APIC'] = APIC(
                        encoding=3,
                        mime='image/jpeg',
                        type=3,
                        desc=u'Cover',
                        data=response.content
                    )
                    audio.save()

            except Exception as e:
                print(f"Error adding cover: {e}")

    def WritingMetaTags(self):
        try:
            # print('[*] FileName : ', self.filename)
            audio = EasyID3(self.filename)
            audio['title'] = self.tags['title']
            audio['artist'] = self.tags['artists']
            audio['album'] = self.tags['album']
            audio['date'] = self.tags['releaseDate']
            audio.save()
            self.setPIC()

        except Exception as e:
            print(f'Error {e}')