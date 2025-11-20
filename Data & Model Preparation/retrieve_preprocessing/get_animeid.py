import requests
import pandas as pd
from bs4 import BeautifulSoup


anime_dict = {}
anime_dict['ID'] = []
anime_dict["Name"] = []
anime_dict['Type'] = []
anime_dict['Episodes'] = []
anime_dict['Aired'] = []
anime_dict['Members'] = []
anime_dict['Scores'] = []


url = "https://myanimelist.net/topanime.php?limit="
for limit in range(0, 7150, 50):

    response = requests.get(url+str(limit))

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        raw_ids = soup.find_all("div", id=lambda x: str(x).startswith("area"))

        ids = [text["id"] for text in raw_ids]
        for id in ids:
            anime_dict["ID"].append(id.split("area")[1])
        # cleaned_ids =  [x.split("area")[1] for x in ids] # get cleaned anime ids
        # print(ids)
        raw_names = soup.find_all("a", class_="hoverinfo_trigger")
        names = []
        for raw in raw_names:
            if raw.text.strip() == '':
                continue
            anime_dict["Name"].append(raw.text.strip())

        raw_info = soup.find_all("div", class_="information")
        # anime_type, aired, members = [], [], []
        for info in raw_info: #get animes information
            lines = list(info.stripped_strings)
            if len(lines) >= 3:
                type = lines[0].split('(')[0].strip()
                episodes = (lines[0].split('(')[1].split(' ')[0])
                anime_dict['Type'].append(type)
                anime_dict['Episodes'].append(episodes)
                anime_dict['Aired'].append(lines[1])
                anime_dict['Members'].append(lines[2])
        #print(raw_info)
        raw_scores = soup.find_all("div", class_="js-top-ranking-score-col")
        for score in raw_scores:
            anime_dict["Scores"].append(score.find("span").text.strip()) # get anime score
        # scores = [div.find("span").text.strip() for div in raw_scores] 
        # print(scores)

        # anime_dict['ID'].extend(cleaned_ids)
        # anime_dict["Name"].extend(names)
        # anime_dict['Type'].extend(anime_type)
        # anime_dict['Aired'].extend(aired)
        # anime_dict['Members'].extend(members)
        # anime_dict['Scores'].extend(scores)
    else: print(f"Failed to retrieve: {response.status_code}")

df = pd.DataFrame.from_dict(anime_dict)

df.to_csv("data/anime_data.csv", index=False)



# print(anime_dict)