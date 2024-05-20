import os
import time
import glob
import math
import json
import base64
import random
import openai
from openai import OpenAI

# set up OpenAI client and API key
client = OpenAI()

# set an env var to store your API key
# export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
openai.api_key = os.getenv("OPENAI_API_KEY")

# RATE LIMITS -----------------------------------------------------------------
# 
# For gpt-4-vision-preview as a Tier-2 user, rate limits are as below:
#
# 	RPM (Rate Per Minute): 		100 call / min
# 	RPD (Rate Per Day):			1,000 calls / day
# 	TPM (Tokens Per Minute):	20,000 tokens / min
#
# The above information was obtained on Jan 22, 2024

# delay is set in seconds

delay_in_seconds = 87 # this would be 1.45 min / call, approx. 993 calls / day

# -----------------------------------------------------------------------------


def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def generate_caption(image_path):
	b64_image = encode_image(image_path)

	try:
		response = client.chat.completions.create(
    			model="gpt-4-vision-preview",
    			messages=[{
				"role": "user",
				"content": [{
					"type": "text", 
					"text": "Please describe the type, style, material, color, and components/feature of the object shown in this image. Your response should be a detailed description of the object without mentioning of the background, in a single, short sentence less than 10 words. "},
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{b64_image}",
					},
				}]}],
			max_tokens=300
		)
	except openai.BadRequestError as e:
		return '400'
	return response.choices[0].message.content



categories = ['chair', 'table']
cat_ids = ['03001627', '04379243']
for cate, cate_id in zip(categories[1:], cat_ids[1:]):

	search_path = f'/data/ting/text_guided_3D_gen/TAPS3D/ShapeNetCoreRendering/img/{cate_id}/**'
	uid_dirs = glob.glob(search_path)

	save_path = f'/data/ting/text_guided_3D_gen/gpt_vision/captions/{cate}'
	# os.mkdirs(save_path, exist_ok=True) # I've created the directory manually, no need to run this
	
	json_path = os.path.join(save_path, 'id_captions.json')
	# open(json_path, 'w').close()
	
	temp_storage = {}
	print(f'len(uid_dirs): {len(uid_dirs)}')
	for i, uid_dir in enumerate(uid_dirs[8210:]):
		
		real_i = i + 8210
		
		uid = uid_dir.split(os.sep)[-1]
		model_dir = os.path.join(uid_dir, 'models')
		img_list = [f for f in os.listdir(model_dir) if f.endswith('png')]
		
		# keep trying until an image is successfully captioned 
		caption = '400'
		while caption == '400':
			img_fn = img_list[random.randint(0, len(img_list)-1)] # a, b in$
			img_fp = os.path.join(model_dir, img_fn)
			caption = generate_caption(img_fp)

		temp_storage[uid] = [caption, cate_id]
		
		# store temp storage into json every N resposnes
		N = 10
		if i % N == 0 or real_i == len(uid_dirs)-1:				
			
			if real_i == N:
				json_str = json.dumps(temp_storage, indent=2)
				with open(json_path, 'w') as f:
					f.write(json_str)

				print(f'>> JSON created with first {N} captions...')
				temp_storage = {}				

			elif len(temp_storage) != N:
				with open(json_path) as f:
					content = f.read()

				data = dict(json.loads(content))
				data.update(temp_storage)
				json_str = json.dumps(data, indent=2)

				open(json_path, 'w').close()

				with open(json_path, 'w') as f:
					f.write(json_str)

				print(f'>> Stored {real_i} captions for category {cate}')
				temp_storage = {}
				
		time.sleep(delay_in_seconds)
		

