from rembg import remove, new_session
from PIL import Image
import os
import pandas as pd

df = pd.read_csv('data/pokemon_resources.csv')['name']
input_path = [f for f in os.listdir('data/images/') if f in df.values]
output_dir = 'data/train/'
session = new_session('isnet-anime')

for path in input_path:
    print(f'Processing: {path}')
    os.makedirs(f'{output_dir}{path}', exist_ok=True)
    for image in os.listdir(f'data/images/{path}'):
        input = Image.open(f'data/images/{path}/{image}')
        output = remove(input, session=session)
        output.save(f'{output_dir}{path}/{os.path.splitext(image)[0]}.png')
