# In this file, we will:
#   - Generate 3D models based on captions produced for CLIP-R Precision
#   - Render those models
#   - Make CLIP prediction on those renders
#   - Calculate CLIP-R Precision

import os
import pickle5 as pickle
import shutil
import logging
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == "__main__":
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	
	parser.add_argument("--chair_checkpoint_dir", default="", type=str, help="Name of directory containing checkpoints for the chair category. This is what you set your outdir to be when you ran the training script for TAPS3D. ")
	parser.add_argument("--chair_checkpoint_name", default="", type=str, help="Name of chair checkpoint to be evaluated across the three caption types. ")
	parser.add_argument("--table_checkpoint_dir", default="", type=str, help="Name of directory containing checkpoints for the table category. This is what you set your outdir to be when you ran the training script for TAPS3D. ")
	parser.add_argument("--table_checkpoint_name", default="", type=str, help="Name of table checkpoint to be evaluated across the three caption types. ")
	args = parser.parse_args()
	
	chair_ckpt_dir = args.chair_checkpoint_dir
	table_ckpt_dir = args.table_checkpoint_dir
	
	chair_ckpt = args.chair_checkpoint_name
	table_ckpt = args.table_checkpoint_name
	
	# you can also manually fill in your checkpoint folders and checkpoints if you gave the folders and checkpoints different names across the three types of captions
	checkpoints = {
	    'pseudo_captions': {'chair': f'{chair_ckpt_dir}/{chair_ckpt}', 'table': f'{table_ckpt_dir}/{chair_ckpt}'},
	    'human_captions': {'chair': f'{chair_ckpt_dir}/{chair_ckpt}', 'table': f'{table_ckpt_dir}/{chair_ckpt}'},
	    'gpt4_captions': {'chair': f'{chair_ckpt_dir}/{chair_ckpt}', 'table': f'{table_ckpt_dir}/{chair_ckpt}'}
	}
	
	cwd = os.path.abspath(os.getcwd())
	clip_r_precision_root = cwd
	taps3d_root = os.path.join(os.path.dirname(cwd), 'TAPS3D')
	pickles_root = os.path.join(clip_r_precision_root, 'pickles')

	finetuned_clip_root = os.path.join(taps3d_root, 'data/finetune_clip/model_checkpoint')

	caption_types = ['pseudo_captions', 'human_captions', 'gpt4_captions']
	categories = ['chair', 'table']
	attributes = ['shapes', 'colors']
	splits = ['test_seen', 'test_unseen', 'test_swapped', 'test_unseen_diverse_style']
	# note: 'swapped_text' is in data_pkl[uid][caption_id]['swapped_text']

	# loop each <caption type>
	for caption_type in caption_types:

		# loop each <model type>
		for category in categories:
		
			# get approppreiate checkpoint for each <caption type> and <model type>
			model_fp = os.path.join(taps3d_root, 'data', caption_type, category, checkpoints[caption_type][category])
			
			# loop caption <attribute> (shape and color)
			for attr in attributes:
				
				# load split.pkl
				split_pkl_fp = os.path.join(pickles_root, caption_type, category, attr, 'split.pkl')
				with open(split_pkl_fp, 'rb') as f:
					split_dict = pickle.load(f)

				for split in splits:
					
					split_prediction_pkl = []
					
					activate_conda_env_cmd = '. ~/anaconda3/etc/profile.d/conda.sh && conda activate taps3d'.split(' ')
					res = subprocess.run(activate_conda_env_cmd, shell=True, capture_output=True, text=True)
					print(f"COMMAND:\n{' '.join(res.args)}")
					print(f"STDERR: {repr(res.stderr)}")
					print(f'STDOUT: {res.stdout}')
					print(f'RETURN CODE: {res.returncode}')
					
					# generate models from checkpoint based on captions in pickle
					if split == 'test_swapped':
					
						data_pkl_fp = os.path.join(pickles_root, caption_type, category, attr, 'data.pkl')
						with open(data_pkl_fp, 'rb') as f:
							data_dict = pickle.load(f)
						
						print('>> entered test_swapped flow')
						
						for uid in split_dict['test_seen']:
							cap_ids = list(data_dict[uid].keys())
							for cap_id in cap_ids:
								if 'swapped_text' not in list(data_dict[uid][cap_id].keys()): 
									continue
								caption = data_dict[uid][cap_id]['swapped_text']
								
								outdir = f'save_inference_results/{caption_type}/{category}/{attr}/{split}'
								if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)
								alt_dir_name = f'{uid}_{cap_id}'
								
								# inference root
								generated = False
								inference_root = os.path.join(taps3d_root, outdir, alt_dir_name)
								if not os.path.exists(inference_root):
									os.makedirs(inference_root, exist_ok=True)
								else:
									if os.path.exists(os.path.join(inference_root, 'None_000000.png')):
										skipped += 1
										# print(f'{skipped} skipped')
										generated = True # skip generating samples if directory exists

								# construct command

								caption = caption.translate({ord(c): None for c in '"'})
								if not generated:
									generate_samples_py = os.path.join(taps3d_root, 'generate_samples.py')
									generate_model_cmd = f'python {generate_samples_py} --network {model_fp} --class_id {category} --seed 0 --outdir {outdir} --inference_to_generate_rendered_img True --n_sample 1 --use_alt_dir_name {alt_dir_name} --text "{caption}"'
								
									with subprocess.Popen(generate_model_cmd, shell=True, stdout=subprocess.PIPE, bufsize=1) as sp:
										for line in sp.stdout:
											logging.critical(line)
								
								# remove texture_mesh_for_inference directory which contains the obj
								if os.path.exists(os.path.join(inference_root, 'texture_mesh_for_inference')):
									shutil.rmtree(os.path.join(inference_root, 'texture_mesh_for_inference'))
								# remove all other .gif and .pngs that we don't need
								for f in os.listdir(inference_root):
									if f == 'None_000000.png': continue
									os.remove(os.path.join(inference_root, f))
									
								# render image path
								render_fp = os.path.join(inference_root, 'None_000000.png')
								assert os.path.exists(render_fp), f'uid: {uid}, cap_id: {cap_id}, None_000000.png not found.'
								
								split_prediction_pkl.append((uid, cap_id, render_fp, 'placeholder caption'))

					else:
						if split == 'test_unseen_diverse_style':
							data_pkl_fp = os.path.join(pickles_root, caption_type, category, attr, 'test_unseen_diverse_style', 'data.pkl')
						else:
							data_pkl_fp = os.path.join(pickles_root, caption_type, category, attr, 'data.pkl')
						
						with open(data_pkl_fp, 'rb') as f:
							data_dict = pickle.load(f)
					
						print(f'>> entered {split} flow')
						skipped = 0
						for uid in split_dict[split]:
							cap_ids = list(data_dict[uid].keys())
							for cap_id in cap_ids:
							
								caption = data_dict[uid][cap_id]['text']
								
								outdir = os.path.join(taps3d_root, 'save_inference_results', caption_type, category, attr, split)
								if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)
								alt_dir_name = f'{uid}_{cap_id}'
								
								# inference root
								inference_root = os.path.join(taps3d_root, outdir, alt_dir_name)
								generated = False
								if not os.path.exists(inference_root):
									os.makedirs(inference_root, exist_ok=True)
								else:
									if os.path.exists(os.path.join(inference_root, 'None_000000.png')):
										skipped += 1
										#print(f'{skipped} skipped')
										generated=True # skip generating samples if directory exists
								
								# construct command

								caption = caption.translate({ord(c): None for c in '"'})


								if not generated:
									generate_samples_py = os.path.join(taps3d_root, 'generate_samples.py')
									generate_model_cmd = f'python {generate_samples_py} --network {model_fp} --class_id {category} --seed 0 --outdir {outdir} --inference_to_generate_rendered_img True --n_sample 1 --use_alt_dir_name {alt_dir_name} --text "{caption}"'
									# generate 1 sample
									with subprocess.Popen(generate_model_cmd, shell=True, stdout=subprocess.PIPE, bufsize=1) as sp:
										for line in sp.stdout:
											logging.critical(line)

								# remove texture_mesh_for_inference directory which contains the obj
								if os.path.exists(os.path.join(inference_root, 'texture_mesh_for_inference')):
									shutil.rmtree(os.path.join(inference_root, 'texture_mesh_for_inference'))
								# remove all other .gif and .pngs that we don't need
								for f in os.listdir(inference_root):
									if f == 'None_000000.png': continue
									os.remove(os.path.join(inference_root, f))
								
								# render image path
								render_fp = os.path.join(inference_root, 'None_000000.png')
								assert os.path.exists(render_fp), f'uid: {uid}, cap_id: {cap_id}, {render_fp} not found.'
								
								split_prediction_pkl.append((uid, cap_id, render_fp, 'placeholder caption'))
					
					# save split_prediction_pkl
					pred_pkl_fp = os.path.join(pickles_root, caption_type, category, attr, split, 'pred_path.pkl')
					with open(pred_pkl_fp, 'wb') as f:
						pickle.dump(split_prediction_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
					
					# find appropriate finetuned CLIP weights (.pt file)
					finetuned_clip_fp = os.path.join(finetuned_clip_root, caption_type, category, 'model_29.pt')
					
					# set up out path for after clip predicted captions are added into pred_path.pkl
					out_pkl_fp = os.path.join(pickles_root, caption_type, category, attr, split, 'clip_pred_path.pkl')
					
					# run clip prediction
					make_clip_prediction_py = os.path.join(clip_r_precision_root, 'make_clip_prediction.py')
					mk_clip_pred_cmd = f'python {make_clip_prediction_py} --dataset custom --comp_type {attr} --split {split} --ckpt {finetuned_clip_fp} --gpu 1 --category {category} --caption_type {caption_type} --pred_path {pred_pkl_fp} --out_path {out_pkl_fp}'
					with subprocess.Popen(mk_clip_pred_cmd, shell=True, stdout=subprocess.PIPE, bufsize=1) as sp:
						for line in sp.stdout:
							logging.critical(line)

					# compute CLIP-R Precision
					print(f'CLIP precision scores for {caption_type} {category} {attr} {split}')
					compute_r_precision_py = os.path.join(clip_r_precision_root, 'compute_r_precision.py')
					compute_r_precision_cmd = f'python {compute_r_precision_py} --dataset custom --comp_type {attr} --split {split} --pred_path {out_pkl_fp}'
					with subprocess.Popen(compute_r_precision_cmd, shell=True, stdout=subprocess.PIPE, bufsize=1) as sp:
						for line in sp.stdout:
							logging.critical(line)

