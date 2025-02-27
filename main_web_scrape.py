import requests

import pandas as pd
import pprint as pp
import datetime as dt
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path
import os

NBA_TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def get_team_names(url: str) -> str:
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    title_text = soup.title.get_text()

    home, away = title_text.split(",")[0].split(" vs ")
    return home, away


def calculate_combined(
    basic: pd.DataFrame, advanced: pd.DataFrame, team_name: str, side: str
) -> pd.DataFrame:
    basic.columns = basic.columns.droplevel(0)
    advanced.columns = advanced.columns.droplevel(0)
    merged = pd.merge(
        left=basic,
        right=advanced,
        on="Starters",
        suffixes=("_basic", "_advanced"),
    )
    merged["team_name"] = team_name
    merged["side"] = side
    merged = merged[merged["Starters"] != "Reserves"]
    return merged


def get_game_statistics(url: str) -> pd.DataFrame:
    dfs = pd.read_html(url)
    home, away = get_team_names(url)

    basic_away = dfs[0]
    advanced_away = dfs[7]

    basic_home = dfs[8]
    advanced_home = dfs[15]

    merged_home = calculate_combined(
        basic_home, advanced_home, team_name=home, side="home"
    )
    merged_away = calculate_combined(
        basic_away, advanced_away, team_name=away, side="away"
    )

    combined = pd.concat([merged_home, merged_away], ignore_index=True)
    print(combined)


def determine_valid_game_url(urls: list[str], valid_urls: list[str]) -> str:
    for url in urls:
        if url in valid_urls:
            return url

    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            valid_urls.append(url)
            update_valid_urls(valid_urls)
            return url
    raise ValueError(f"None of the following urls are valid: {urls}")


def get_opponent_name(team_name: str) -> str:
    try:
        return NBA_TEAM_ABBREVIATIONS[team_name]
    except KeyError:
        raise NotImplementedError(f"Unsupported team: {team_name}")


def get_valid_urls() -> list[str]:
    path = Path(".") / "valid_urls.json"
    if not path.exists():
        with open(path, "w") as fp:
            json.dump([], fp)
        return []

    with open(path, "r") as fp:
        valid_urls = json.load(fp)
        return valid_urls


def update_valid_urls(valid_urls: list[str]) -> None:
    path = Path(".") / "valid_urls.json"
    if not path.exists():
        with open(path, "w") as fp:
            json.dump([], fp)
        return []

    with open(path, "w") as fp:
        json.dump(valid_urls, fp)


def get_regular_season_games_from_internet(url: str) -> pd.DataFrame:
    games = pd.read_html(url)
    regular_season_games = games[0]
    regular_season_games = regular_season_games[
        regular_season_games["G"] != "G"
    ].set_index("G")
    return regular_season_games


def get_games(game_type: str, url: str) -> pd.DataFrame:
    all_games_path = Path(".") / game_type
    if not all_games_path.exists():
        os.mkdir(all_games_path)
        return get_regular_season_games_from_internet(url)

    game_index = url.split(".html")[0].split("/")[-1]
    url_games_path = all_games_path / f"{game_index}.parquet"

    try:
        games = pd.read_parquet(url_games_path)
        return games
    except FileNotFoundError:
        games = get_regular_season_games_from_internet(url)
        games.to_parquet(url_games_path)
        return games


def get_regular_season_urls(team: str, url: str) -> list[str]:
    valid_urls = get_valid_urls()

    regular_season_games = get_games("regular_season_games", url)

    games = []
    for _, row in regular_season_games.iterrows():
        game_date = dt.datetime.strptime(row["Date"], "%a, %b %d, %Y")

        game_date_str = game_date.strftime("%Y%m%d")

        team_url = team
        try:
            opponent_url = get_opponent_name(row["Opponent"])
        except NotImplementedError:
            pp.pprint(row["Opponent"])
            raise

        game_url_team_home = f"https://www.basketball-reference.com/boxscores/{game_date_str}0{team_url}.html"
        game_url_opponent_away = f"https://www.basketball-reference.com/boxscores/{game_date_str}0{opponent_url}.html"

        try:
            game_url = determine_valid_game_url(
                urls=[game_url_team_home, game_url_opponent_away], valid_urls=valid_urls
            )
        except ValueError:
            pp.pprint(row)
            exit()
        games.append(game_url)

        time.sleep(1)

    return games


def main() -> None:
    url = "https://www.basketball-reference.com/boxscores/202310250NYK.html"
    url = "https://www.basketball-reference.com/boxscores/202310270BOS.html"

    # get_game_statistics(url=url)

    url = "https://www.basketball-reference.com/teams/BOS/2024_games.html"
    game_urls = get_regular_season_urls("BOS", url)
    pp.pprint(game_urls)


if __name__ == "__main__":
    main()
