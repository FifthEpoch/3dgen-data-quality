"""
This code produces:
	>> split.pkl: contains lists of image ids (uids or uris) for train, test_seen, and test_unseen. It also contains information about the heldout pairs in <adj>_<nn> formatted string as a set
	>> data.pkl: contains original caption and swqapped captions retrievable by image ids
	>> pred_path.pkl: contains [(image_id, caption_id, generated_image_path, DAMSM_prediction_placeholder), ...] for running make_clip_prediction.py
The formats of these outputs are compatible with the original CLIP-R Precision implementation comp-t2i-dataset found here: https://github.com/Seth-Park/comp-t2i-dataset/tree/main
"""

import os
import json
import spacy
import pickle
import random
import numpy as np
from pathlib import Path

# /home/ptclient/text_guided_3D_gen/comp-t2i-dataset/pickles/human_captions/chair/colors/test_unseen/pred_path.pkl

caption_types = ['human_captions', 'pseudo_captions', 'gpt4_captions']
model_types = ['chair', 'table']
attrs = ['colors', 'shapes']
split_types = ['train', 'test_seen', 'test_unseen', 'test_swapped']

nlp = spacy.load('en_core_web_sm')

unseen_dict = {}
for model_type in model_types:
    unseen_dict[model_type] = {}
    for attr in attrs:
        unseen_dict[model_type][attr] = {}
        for caption_type in caption_types:
            unseen_dict[model_type][attr][caption_type] = []

for caption_type in caption_types:
    for model_type in model_types:
        caption_fp = f'/home/ptclient/text_guided_3D_gen/TAPS3D/data/{caption_type}/{model_type}/id_captions.json'
        with open(caption_fp, 'r') as f:
            caption_data = json.load(f)
        for attr in attrs:
            print(f'{caption_type}, {model_type}, {attr}')

            pickle_root = f'/home/ptclient/text_guided_3D_gen/comp-t2i-dataset/pickles/{caption_type}/{model_type}/{attr}'

            with open(f'/home/ptclient/text_guided_3D_gen/comp-t2i-dataset/data/{attr}.txt', 'r') as f:
                comp_type_adjs = f.read().split('\n')

            # set up adjective-noun frequencies tracker
            # first level key: 
            # 	>> <adjective>_<lemmatized noun>
            # first level value: 
            #	>> list of dict objects
            # second level keys: 
            # 	1. uid: 			model unique id in ShapeNet
            # 	2. caption id: 		0-based index converted to str
            # 	3. adj: 			original casing
            # 	4. nn: 				original casing, before lemmatization
            adj_nn_freq = {}

            # first level key: adjective, lowercase
            # value: frequencies
            adj_freq = {}

            data_pkl = {}

            all_pred_path = []

            # assign uids to train or test_unseen

            # loop through the captions
            uids = caption_data.keys()
            print(f'len(uids): {len(uids)}')
            for uid in uids:
                # CONFIG HERE: make sure variable caption is a string that contains a single caption
                model_data = caption_data[uid]
                caption = model_data[0]
                doc = nlp(caption)

                # store in data_pkl
                if uid not in data_pkl.keys():
                    data_pkl[uid] = {}
                caption_id = len(data_pkl[uid])
                data_pkl[uid][caption_id] = {'text': caption}

                # now we save the necessary information in a tuple for pred_path_pkl
                # the pickle file should contain the following content according to the authors
                # >> [(image_id, caption_id, generated_image_path, DAMSM_prediction), ...]
                if model_type == 'chair':
                    category_id = '03001627'
                else: # table
                    category_id = '04379243'
                img_fn = str(random.randint(0, 23)).zfill(3) + '.png'
                img_path = os.path.join('/home/ptclient/text_guided_3D_gen/TAPS3D/ShapeNetCoreRendering/img', category_id, uid, 'models', img_fn)
                all_pred_path.append((uid, caption_id, img_path, 'PLACEHOLDER TEXT'))

                # find adj-noun pairs by frequencies
                for i, token in enumerate(doc):
                    if i == len(doc)-1: continue 			# make sure token is adjective
                    if token.pos_ != 'ADJ': continue		# make sure token has next
                    next_token = doc[i+1]
                    if next_token.pos_ != 'NOUN': continue	# make sure next token is noun
                    # standardize adjective with lowercase convertion
                    adj = token.text.lower()

                    # track adjective frequencies
                    if adj not in adj_freq.keys():
                        adj_freq[adj] = 1
                    else:
                        adj_freq[adj] += 1

                    # standardize noun by lemmatization and lowercase convertion
                    lemma_noun = next_token.lemma_

                    adj_nn_key = adj + '_' + lemma_noun
                    if adj_nn_key not in adj_nn_freq.keys():
                        adj_nn_freq[adj_nn_key] = [{
                            'uid': uid,
                            'caption_id': caption_id,
                            'adj': token.text,
                            'nn': next_token.text}]
                    else:
                        adj_nn_freq[adj_nn_key].append({
                            'uid': uid,
                            'caption_id': caption_id,
                            'adj': token.text,
                            'nn': next_token.text})

            # determine heldout pairs
            adj_nn_freq_list = sorted(adj_nn_freq, key=lambda k: len(adj_nn_freq[k]), reverse=True)
            percentile_25 = int((len(adj_nn_freq_list) / 100) * 25)
            percentile_75 = int((len(adj_nn_freq_list) / 100) * 75) + 1
            heldout_pairs = adj_nn_freq_list[percentile_25:percentile_75]
            heldout_pairs_num = max(100, int(len(adj_nn_freq_list) / 10))
            print(f'heldout_pairs_num: {heldout_pairs_num}, len(adj_nn_freq_list)/10: {int(len(adj_nn_freq_list) / 10)}')
            heldout_pairs = random.sample(heldout_pairs, heldout_pairs_num)
            print(f'len(adj_nn_freq_list): {len(adj_nn_freq_list)}')
            print(f'len(heldout_pairs): {len(heldout_pairs)}')

            # create a list of test_unseen uids
            test_unseen = set()
            # add heldout_pairs key:value pair to data_pkl
            for pair in heldout_pairs:
                occurrences = adj_nn_freq[pair]
                for o in occurrences:
                    uid = o['uid']
                    caption_id = o['caption_id']

                    test_unseen.add(uid)

                    if 'heldout_pairs' not in data_pkl[uid][caption_id].keys():
                        data_pkl[uid][caption_id]['heldout_pairs'] = [pair]
                    else:
                        data_pkl[uid][caption_id]['heldout_pairs'].append(pair)

            # create training set by excluding test_unseen
            train = set(caption_data.keys()).difference(test_unseen)
            # create test seen by making sure test_seen and test_unseen are the same size
            test_seen = set(random.sample(list(train), len(test_unseen)))

            # configure split.pkl
            split_pkl = {
                'train': list(train), 
                'test_seen': list(test_seen), 
                'test_unseen': list(test_unseen), 
                'heldout_pairs': set(heldout_pairs)
            }
            
            # save unseen uids to unseen dict
            unseen_dict[model_type][attr][caption_type] = split_pkl['test_unseen']

            # find the 60 most frequent adjectives
            top_60_adjs = sorted(adj_freq, key=lambda k: adj_freq[k], reverse=True)
            # filter out adjectives that aren't in color or shape adjectives list
            top_60_adjs = [adj for adj in top_60_adjs if adj in comp_type_adjs]

            # find top 100 most frequent adjective-noun pairs based on the top 60 adjectives
            top_100_pairs = {}
            for i in range(100):
                top_100_pairs[f'<adj_nn_{i}>'] = 0
            for adj_nn_key in adj_nn_freq_list:
                if adj_nn_key.split('_')[0] not in top_60_adjs: continue
				
                # get frequency of this adj_nn pair
                this_adj_nn_freq = len(adj_nn_freq[adj_nn_key])
                		
                # sort from high frequency to low frequency
                top_100_pairs_sorted_keys = sorted(top_100_pairs, key=top_100_pairs.get, reverse=True)
                		
                # check if it is higher than the lowest frequency pair in top_100
                if top_100_pairs[top_100_pairs_sorted_keys[-1]] < this_adj_nn_freq:
                    del top_100_pairs[top_100_pairs_sorted_keys[-1]]
                    top_100_pairs[adj_nn_key] = this_adj_nn_freq
            	
            # make sure top 100 pairs does not contain any 0 frequency
            new_top_100_pairs = {}
            for k, v in top_100_pairs.items():
                if v != 0:
                    new_top_100_pairs[k] = v
            top_100_pairs = new_top_100_pairs
            
            # sort one more time from high frequency to low frequency
            top_100_pairs_sorted_keys = sorted(top_100_pairs, key=top_100_pairs.get, reverse=True)

            # we first identify the dorminant pairs in the top 100 adjective-noun pairs
            # by selecting the pairs in the 25th percentile of top_100_pairs
            if len(top_100_pairs) == 100:
                top_100_25th_percentile_freq = np.percentile(list(top_100_pairs.values()), 25)
            elif len(top_100_pairs) > 85: # in case of insufficient number of heldout pairs
                top_100_25th_percentile_freq = np.percentile(list(top_100_pairs.values()), 35)
            else: # in case of extremely insufficient number of heldout pairs
                top_100_25th_percentile_freq = np.percentile(list(top_100_pairs.values()), 45)
            top_100_above_25th = []
            for key in list(top_100_pairs.keys()):
                if top_100_pairs[key] > top_100_25th_percentile_freq:
                    top_100_above_25th.append(key)

            # for swapping adjectives
            heldout_pairs_adj = [pair.split('_')[0] for pair in heldout_pairs]
            # keep track of swapped
            test_swapped_list = []
            # we produce swapped captions from test_seen
            print(f'len(test_seen): {len(test_seen)}')
            for uid in test_seen:
                caption_objs = data_pkl[uid]
                for caption_id in caption_objs.keys():
                    ori_caption = caption_objs[caption_id]['text']
                    	
                    # find adjective-noun pairs
                    adj_nn_pairs = []
                    adj_nn_pairs_freq = []
                    ori_adjs = []
                    doc = nlp(ori_caption)
                    for i, token in enumerate(doc):
                        if i == len(doc)-1: continue 			# make sure token is adjective
                        if token.pos_ != 'ADJ': continue		# make sure token has next
                        next_token = doc[i+1]
                        if next_token.pos_ != 'NOUN': continue		# make sure next token is noun
					
                        # standardize adjective with lowercase convertion
                        adj = token.text.lower()
                        lemma_noun = next_token.lemma_
						
                        ori_adjs.append(token.text)
                        adj_nn_pairs.append(adj + '_' + lemma_noun)
                        	
                    if len(adj_nn_pairs) == 0:
                        continue
                    elif len(adj_nn_pairs) == 1:
                        ori_adj = ori_adjs[0].lower()
                        new_adj = ori_adj
                        while new_adj == ori_adj:
                            new_adj = heldout_pairs_adj[random.randint(0, len(heldout_pairs_adj)-1)]
                        data_pkl[uid][caption_id]['swapped_text'] = ori_caption.replace(ori_adjs[0], new_adj)
                        data_pkl[uid][caption_id]['changes_made'] = {
                            'noun': adj_nn_pairs[0].split('_')[-1], 
                            'original_adj': ori_adj,
                            'new_adj': new_adj
                        }
                    else: # more than one option for swapping

                        # check if adj_nn_pairs in 25th percentile of top_100_pairs
                        if set(adj_nn_pairs).issubset(top_100_above_25th):
                            # adj_nn_pairs is a subset of top_100_above_25th
                            # every option is in the 25th percentile
                            # no adjustments can be made
                            noun = adj_nn_pairs[0].split('_')[-1]
                            ori_adj = ori_adjs[0].lower()
                            new_adj = ori_adj
                            while new_adj == ori_adj:
                                new_adj = heldout_pairs_adj[random.randint(0, len(heldout_pairs_adj)-1)]
                        else:

                            ori_adj = ori_adjs[0].lower()

                            # find the option that is not in 25th percentile of top_100_pairs
                            adj_noun_candidates = []
                            for pair in adj_nn_pairs:
                                if pair not in top_100_above_25th:
                                    adj = pair.split('_')[0]
                                    if adj != ori_adj:
                                        adj_noun_candidates.append(adj + '_' + pair.split('_')[-1])

                            # in case all pairs not in top_100_above_25th have the ori_adj
                            if len(adj_noun_candidates) == 0:
                                adj_noun_candidates = adj_nn_pairs

                            adj_noun_candidate = adj_noun_candidates[random.randint(0, len(adj_noun_candidates)-1)]

                            new_adj = ori_adj
                            while new_adj == ori_adj:
                                new_adj = heldout_pairs_adj[random.randint(0, len(heldout_pairs_adj)-1)]
                            noun = adj_noun_candidate.split('_')[1]

                        data_pkl[uid][caption_id]['swapped_text'] = ori_caption.replace(ori_adjs[0], new_adj)
                        data_pkl[uid][caption_id]['changes_made'] = {
                            'noun': noun,
                            'original_adj': ori_adj,
                            'new_adj': new_adj
                        }
                    print(f"swap made at {uid}_{caption_id}: {data_pkl[uid][caption_id]['changes_made']['original_adj']} -> {data_pkl[uid][caption_id]['changes_made']['new_adj']} {data_pkl[uid][caption_id]['changes_made']['noun']}")
                    test_swapped_list.append(uid)
            
            # save split.pkl
            # configure split.pkl
            split_pkl = {
                'train': list(train),
                'test_seen': list(test_seen),
                'test_unseen': list(test_unseen),
                'test_swapped': test_swapped_list,
                'heldout_pairs': set(heldout_pairs)
            }

            # save split.pkl
            with open(os.path.join(pickle_root, 'split.pkl'), 'wb') as f:
                pickle.dump(split_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

            pred_path_pkl = {split: [] for split in split_types}
            for entry in all_pred_path:
                for split in split_types:
                    if entry[0] in split_pkl[split]:
                        pred_path_pkl[split].append(entry)

            for split in split_types:
                # save pred_path_pkl
                pkl_save_path = os.path.join(pickle_root, split)
                Path(pkl_save_path).mkdir(parents=True, exist_ok=True) # create any missing directories
                with open(os.path.join(pkl_save_path, 'pred_path.pkl'), 'wb') as f:
                    pickle.dump(pred_path_pkl[split], f, protocol=pickle.HIGHEST_PROTOCOL)

            # save split.pkl
            with open(os.path.join(pickle_root, 'split.pkl'), 'wb') as f:
                pickle.dump(split_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

            # save data.pkl
            with open(os.path.join(pickle_root, 'data.pkl'), 'wb') as f:
                pickle.dump(data_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
            
# unseen_dict[model_type][attr][caption_type]
            
# create 'test_unseen_diverse_style'
for caption_type in caption_types:
    for model_type in model_types:
        for attr in attrs:

            # load uid info for each split based on caption type, model type, and attribute to be tested
            pickle_root = f'/home/ptclient/text_guided_3D_gen/comp-t2i-dataset/pickles/{caption_type}/{model_type}/{attr}'
            with open(os.path.join(pickle_root, 'split.pkl'), 'rb') as f:
                split_pkl = pickle.load(f)

            # define number of captions used from each cap type to compose the test_unseen_diverse_style split
            unseen_split_len = len(split_pkl['test_unseen'])
            if unseen_split_len % 3 == 0:
                num_cap_from_split = {c: int(unseen_split_len/3) for c in caption_types}
            else:
                rand_int = random.randint(0, 3)
                num_cap_from_split = {c: 0 for c in caption_types}
                for i, c in enumerate(caption_types):
                    num_cap_from_split[c] = int(unseen_split_len/3)
                    if i == rand_int: num_cap_from_split[c] += int(unseen_split_len % 3)

            print(f'num_cap_from_split: {num_cap_from_split}')

            # populating 'test_unseen_diverse_style' with randomly sampled uids
            test_unseen_div_sty = []
            test_unseen_cap_type_dict = {c: [] for c in caption_types}
            for c in caption_types:
                unseen_uids = unseen_dict[model_type][attr][c]
                sample = random.sample(unseen_uids, num_cap_from_split[c])
                test_unseen_div_sty.extend(sample)
                test_unseen_cap_type_dict[c] = sample

            # make pred_path based on test_unseen_cap_type_dict
            pred_path = []
            new_split_data_pkl = {}
            
            for c, uids in test_unseen_cap_type_dict.items():
                for uid in uids:
                
                    # load data.pkl of caption type
                    c_pickle_root = f'/home/ptclient/text_guided_3D_gen/comp-t2i-dataset/pickles/{c}/{model_type}/{attr}'
                    with open(os.path.join(c_pickle_root, 'data.pkl'), 'rb') as f:
                        data_pkl = pickle.load(f)
                        
                    # get caption from uid and a random cap_id
                    cap_id = random.randint(0, len(data_pkl[uid])-1)
                    caption = data_pkl[uid][cap_id]['text']
                
                    # store caption in new_split_data_pkl
                    if uid not in new_split_data_pkl.keys():
                        new_split_data_pkl[uid] = {}
                    caption_id = len(new_split_data_pkl[uid])
                    new_split_data_pkl[uid][caption_id] = {'text': caption}
                    
                    entry = (uid, cap_id, 'IMG PATH', 'PREDICTION_PLACEHOLDER')
                    pred_path.append(entry)

            pkl_save_path = os.path.join(pickle_root, 'test_unseen_diverse_style')
            Path(pkl_save_path).mkdir(parents=True, exist_ok=True) 
            
            with open(os.path.join(pkl_save_path, 'pred_path.pkl'), 'wb') as f:
                pickle.dump(pred_path, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open(os.path.join(pkl_save_path, 'data.pkl'), 'wb') as f:
                pickle.dump(new_split_data_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

            split_pkl['test_unseen_diverse_style'] = test_unseen_div_sty
            with open(os.path.join(pickle_root, 'split.pkl'), 'wb') as f:
                pickle.dump(split_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            
