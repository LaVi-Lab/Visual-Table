import pandas as pd
import io
import base64
import json
from PIL import Image
from tqdm import tqdm
import os
import argparse

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def main(args):
    mmbench_imageID = set() # for mmbench

    datas = pd.read_csv(args.data_path, sep='\t')
    print(datas.shape[0])
    print(datas.columns)
    # 4377
    # Index(['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'answer', 'category',
    #        'image', 'source', 'l2-category', 'comment', 'split'],
    #       dtype='object')

    image_path = args.image_path
    os.makedirs(image_path, exist_ok=True)

    for idx in tqdm(range(len(datas))):
        data = datas.iloc[idx]
        index = int(data['index'])
        imageID = f"{index}.jpg"
        if imageID not in mmbench_imageID:
            mmbench_imageID.add(imageID)
            image = decode_base64_to_image(data['image'])
            image.save(f"{image_path}/{imageID}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./playground/data/eval/mmbench/mmbench_dev_20230712.tsv")
    parser.add_argument("--image_path", type=str, default="./playground/data/eval/mmbench/images/mmbench_dev_20230712")
    args = parser.parse_args()
    main(args)
