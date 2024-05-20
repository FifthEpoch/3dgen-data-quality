"""
A script that reformats json data into format compatiable with TAPS3D
"""

import os
import json


def get_json_data(_fp):
    f = open(_fp)
    data = json.load(f)
    f.close()
    return data

def save_json_file(_dst_dir, _cat, _dict, _dst_fn='all_captions.json'):
    if not os.path.isdir(_dst_dir): os.mkdir(_dst_dir)
    json_fp = os.path.join(_dst_dir, _cat, _dst_fn)
    with open(json_fp, "w") as outf:
        json.dump(_dict, outf)




# CHANGE THESE FOR DIFFERENT CASES -----------------------------------

caption_type = 'human_captions'
root = './'
dst_root = os.path.join(root, f'{caption_type}')

# --------------------------------------------------------------------


ps_chair_data = get_json_data(os.path.join(root, 'pseudo_captions', 'chair', 'id_captions.json'))
ps_table_data = get_json_data(os.path.join(root, 'pseudo_captions', 'table', 'id_captions.json'))
chair_ids = ps_chair_data.keys()
table_ids = ps_table_data.keys()

src_data = get_json_data(os.path.join(root, f'{caption_type}_shapenet.json'))

formatted_chair = {}
formatted_table = {}


chair_all_captions = []
table_all_captions = []

for i in range(len(src_data['captions'])):
    synset_id = src_data["captions"][i]["model"]
    caption = " ".join(src_data["captions"][i]["caption"]).replace(" .", ".")

    # find out if model's id appears in table or chair set
    model_type = None
    if synset_id in chair_ids: model_type = '03001627'
    if synset_id in table_ids: model_type = '04379243'
    # assert model_type, 'Model does not belong to chair or table category.'

    if model_type == '03001627':
        formatted_chair[synset_id] = [caption, model_type]
        chair_all_captions.append(caption)
    elif model_type == '04379243':
        formatted_table[synset_id] = [caption, model_type]
        table_all_captions.append(caption)

save_json_file(dst_root, 'chair', formatted_chair)
save_json_file(dst_root, 'table', formatted_table)









