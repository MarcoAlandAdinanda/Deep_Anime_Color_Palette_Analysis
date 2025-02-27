import pandas as pd
import requests
from bs4 import BeautifulSoup

years = [2023, 2024]
seasons = ['winter', 'fall']#['winter', 'spring', 'summer', 'fall']

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070213 BonEcho/2.0.0.2pre'
}

### GET TITLES
for year in years:
    for season in seasons:
        container_anime = []
        container_character = []

        baseurl = f"https://myanimelist.net/anime/season/{year}/{season}"
        r = requests.get(baseurl)
        soup = BeautifulSoup(r.content, 'lxml')

        anime_page = soup.find_all('div', class_='js-anime-category-producer seasonal-anime js-seasonal-anime js-anime-type-all js-anime-type-1')

        ### GET ANIME INFO
        for anime in anime_page:
            anime_link = anime.find_all('a', href=True)[0]['href']

            r_anime = requests.get(anime_link, headers=headers)
            soup_anime = BeautifulSoup(r_anime.content, 'lxml')

            ### AS INDEX
            jp_title = soup_anime.find('h1', class_='title-name h1_bold_none').text.strip()
            try:
                en_title = soup_anime.find('p', class_='title-english title-inherit').text.strip()
            except:
                en_title = None
            

            box_anime = {}
            box_anime['year'] = year
            box_anime['season'] = season
            box_anime['jp_title'] = jp_title
            box_anime['en_title'] = en_title

            ### ANIME_ATTRIBUTES
            # leftside = soup_anime.find('div', class_='leftside')
            # try:
            #     img = leftside.find('img')['data-src']
            # except:
            #     img = None
            # box_anime['img'] = img

            # for pad in leftside.find_all('div', class_='spaceit_pad'):
            #     try:
            #         key = pad.text.strip()
            #         value = pad['href'].strip()
            #     except:
            #         text = pad.text.split(':')
            #         key, value = text[0].strip(), text[1].strip()
                
            #     box_anime[key] = value

            ### CHARACTER ATTRIBUTES
            for char in soup_anime.find_all('h3', class_='h3_characters_voice_actors'):
                box_character = {}
                box_character['jp_title'] = jp_title
                box_character['en_title'] = en_title

                char_link = char.find('a', href=True)['href']
                
                r_char = requests.get(char_link, headers=headers)
                soup_char = BeautifulSoup(r_char.content, 'lxml')

                name = soup_char.find('h2', "normal_header").text
                fav = soup_char.find('td', class_='borderClass').text.strip()
                fav = fav[fav.find('Member Favorites'):]
                try:
                    img = soup_char.find('img', class_='portrait-225x350')['data-src']
                except:
                    img = None

                box_character['char_name'] = name
                box_character['fav'] = fav
                box_character['img'] = img

                container_character.append(box_character)

            # container_anime.append(box_anime)
            print('[SAVING]', jp_title)
        

        # df_anime = pd.DataFrame(container_anime)
        df_char = pd.DataFrame(container_character)

        # df_anime.to_csv(f"anime_{season}_{year}.csv")
        df_char.to_csv(f"char_{season}_{year}.csv")
        
        print(f"[EXPORTING] {season}-{year} dataset")