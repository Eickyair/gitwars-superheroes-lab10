import requests
import pandas as pd
import re
import os
import re


def convert_height_to_cm(height_list,verbose=False):
    """
    Convert height to centimeters from various formats.
    height_list is expected to be a list with two elements (e.g., ["6'2", "188 cm"])
    """
    if not height_list or len(height_list) == 0:
        return None
    def cm_handler(x):
        x = x.lower()
        if not 'cm' in x:
            return None
        regex_float = r'\d+\.\d+'
        match = re.search(regex_float, x)
        if match:
            return float(match.group(0))
        regex_int = r'\d+'
        match = re.search(regex_int, x)
        if match:
            return float(match.group(0))

    def feet_inches_handler(x):
        x = x.lower()

        # Match formats like "6'2", "5'11", "6 ft 2 in", "6 feet 2 inches"
        match = re.search(r"(\d+)\s*'?[\sft]*\s*(\d+)?\s*['\"in]*", x)
        if match:
            feet = float(match.group(1))
            inches = float(match.group(2)) if match.group(2) else 0
            return (feet * 12 + inches) * 2.54
        # Match formats like "2 m" or "1.85 meters"
        m_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?:eters)?', x)
        if m_match:
            return float(m_match.group(1)) * 100
        return None

    handlers = [feet_inches_handler, cm_handler]
    for height in height_list:
        firstTrueHandler = next((handler(height) for handler in handlers if handler(height) is not None), None)
        if firstTrueHandler is not None:
            if verbose:
                print(f'From {height} to {firstTrueHandler} cm')
            return firstTrueHandler

    raise ValueError(f"No valid height format found: {height_list}")


def convert_weight_to_kg(weight_list, verbose=False):
    """
    Convert weight to kilograms from various formats.
    weight_list is expected to be a list with two elements (e.g., ["185 lb", "84 kg"])
    """
    if not weight_list or len(weight_list) == 0:
        return None

    def lb_weight_handler(x):
        x = x.lower()
        # Match formats like "185 lb", "185 lbs"
        match = re.search(r'(\d+(?:\.\d+)?)\s*lb', x)
        if match:
            return float(match.group(1)) * 0.453592
        return None
    def kg_weight_handler(x):
        x = x.lower()
        match = re.search(r'(\d+(?:\.\d+)?)\s*kg', x)
        if match:
            return float(match.group(1))
        return None
    handlers = [lb_weight_handler, kg_weight_handler]

    for weight in weight_list:

        firstValidHandler = next((handler(weight) for handler in handlers if handler(weight) is not None), None)
        if firstValidHandler is not None:
            if verbose:
                print(f'From {weight} to {firstValidHandler} kg')
            return firstValidHandler
    raise ValueError(f"No valid weight format found: {weight_list}")

def process_basic_hero(hero, verbose=False):
    """
    Procesa las variables básicas de un héroe (sin power).
    """
    try:
        powerstats = hero.get('powerstats', {})
        intelligence = powerstats.get('intelligence')
        strength = powerstats.get('strength')
        speed = powerstats.get('speed')
        durability = powerstats.get('durability')
        combat = powerstats.get('combat')

        appearance = hero.get('appearance', {})
        height_raw = appearance.get('height', [])
        weight_raw = appearance.get('weight', [])

        height_cm = convert_height_to_cm(height_raw)
        weight_kg = convert_weight_to_kg(weight_raw)

        variables = [intelligence, strength, speed, durability, combat, height_cm, weight_kg]
        if all(v is not None for v in variables):
            result = {
                'intelligence': intelligence,
                'strength': strength,
                'speed': speed,
                'durability': durability,
                'combat': combat,
                'height_cm': height_cm,
                'weight_kg': weight_kg
            }
            result_df = pd.DataFrame([result])
            for col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df = result_df[
                (result_df['height_cm'] > 0) &
                (result_df['weight_kg'] > 0)
            ]
            if verbose:
                print(f"Procesado básico: {result}")
            return result_df
        else:
            if verbose:
                print(f"Datos básicos faltantes para {hero.get('name', 'unknown')}")
            return None
    except Exception as e:
        if verbose:
            print(f"Error procesando básico {hero.get('name', 'unknown')}: {e}")
        return None


def process_hero_power(hero, verbose=False):
    """
    Procesa la variable power de un héroe.
    """
    try:
        powerstats = hero.get('powerstats', {})
        power = powerstats.get('power')

        if power is not None:
            result = {'power': power}
            result_df = pd.DataFrame([result])
            result_df['power'] = pd.to_numeric(result_df['power'], errors='coerce')
            if verbose:
                print(f"Procesado power: {result}")
            return result_df
        else:
            if verbose:
                print(f"Power faltante para {hero.get('name', 'unknown')}")
            return None
    except Exception as e:
        if verbose:
            print(f"Error procesando power {hero.get('name', 'unknown')}: {e}")
        return None


def process_raw_hero(hero, verbose=False):
    """
    Procesa un elemento crudo de la respuesta de la API combinando las variables básicas y power.
    """
    basic_df = process_basic_hero(hero, verbose=verbose)
    power_df = process_hero_power(hero, verbose=verbose)

    if basic_df is not None and power_df is not None and not basic_df.empty and not power_df.empty:
        result_df = pd.concat([basic_df.reset_index(drop=True), power_df.reset_index(drop=True)], axis=1)
        return result_df
    else:
        if verbose:
            print(f"No se pudo procesar completamente {hero.get('name', 'unknown')}")
        return None
def fetch_superhero_data(verbose=False):
    """
    Consume la SuperHero API, procesa las variables requeridas
    y genera data/data.csv con el dataset final.
    """
    print("Fetching superhero data from API...") if verbose else None

    # Fetch data from the API
    api_url = "https://akabab.github.io/superhero-api/api/all.json"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching data from API: {e}") if verbose else None
        return

    # Process the data
    processed_records = []

    for hero in data:
        adapted_hero = process_raw_hero(hero, verbose=False)
        if adapted_hero is None:
            continue
        processed_records.append(adapted_hero)

    df = pd.concat(processed_records, ignore_index=True)



    # Ensure we have exactly 600 records through resampling
    if len(df) > 600:
        df = df.head(600)
        print(f"Truncated to 600 records") if verbose else None
    elif len(df) < 600:
        print(f"Only {len(df)} valid records available. Resampling to reach 600 records...") if verbose else None
        # Resample with replacement to reach exactly 600 records
        df = df.sample(n=600, replace=True, random_state=42).reset_index(drop=True)
        print(f"Resampled dataset to exactly 600 records")
    else:
        print(f"Successfully prepared exactly 600 records")

    os.makedirs('data', exist_ok=True)
    output_path = 'data/data.csv'
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Dataset saved to {output_path}")
        print("\nDataset summary:")
        print(df.head())
    return df


if __name__ == "__main__":
    df = fetch_superhero_data(verbose=True)