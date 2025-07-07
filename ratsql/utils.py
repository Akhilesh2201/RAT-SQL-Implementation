# Load Spider Dataset 

import os
import json
import zipfile
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

zip_path = "/content/drive/MyDrive/spider_data.zip"
extract_dir = "/content/spider"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.startswith("spider_data/") and not member.endswith("/"):
                rel_path = os.path.relpath(member, "spider_data")
                out_path = os.path.join(extract_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(zip_ref.read(member))
    print("Extracted spider_data →", extract_dir)
else:
    print("Already extracted.")

# Load helper
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

spider_dir = "/content/spider"
db_dir = os.path.join(spider_dir, "database")

train_data = load_json(os.path.join(spider_dir, "train_spider.json"))
dev_data = load_json(os.path.join(spider_dir, "dev.json"))
table_schemas = load_json(os.path.join(spider_dir, "tables.json"))
schema_dict = {schema['db_id']: schema for schema in table_schemas}

print(f"Loaded {len(train_data)} training examples")
print(f"Loaded {len(dev_data)} dev examples")
print(f"Loaded {len(schema_dict)} schema definitions")