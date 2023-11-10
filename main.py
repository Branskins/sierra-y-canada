import os
import requests
import pandas as pd


def get_pokemon_resources():
    resources = pd.DataFrame()

    res = requests.get('https://pokeapi.co/api/v2/pokemon?limit=200')
    json_data = res.json()

    if res.status_code == requests.codes.ok:
        resources = pd.DataFrame(json_data['results'])
        while json_data['next'] is not None:
            next_request = json_data['next']
            res = requests.get(next_request)
            json_data = res.json()
            if res.status_code == requests.codes.ok:
                resources = pd.concat([resources, pd.DataFrame(json_data['results'])], ignore_index=True)

    return resources


def save_pokemon_resources(resources):
    os.makedirs('data', exist_ok=True)
    resources.to_csv('data/pokemon_resources.csv', index=False)


def read_pokemon_resources():
    return pd.read_csv('data/pokemon_resources.csv')


def main():
    resources = get_pokemon_resources()
    save_pokemon_resources(resources)


if __name__ == '__main__':
    main()
