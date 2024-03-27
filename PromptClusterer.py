#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datasets import load_dataset
import random
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
from diffusers import AutoencoderTiny
import torch
import numpy as np
import lunar_tools as lt
from threading import Thread
import time
from datetime import datetime
from PIL import Image
import lunar_tools as lt
from tqdm import tqdm

#%%# Load embdedding model

from transformers import AutoModel
from numpy.linalg import norm

cos_sim = lambda a,b: (a @ b.T)[:,0] / (norm(a,axis=1)*norm(b,axis=1))
cos_sim_mtx = lambda a,b: (a @ b.T) / (norm(a,axis=1,keepdims=True)*norm(b,axis=1,keepdims=True).T)
embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
embedding_model.to("cuda")
embeddings = embedding_model.encode(['How is the weather today?', 'What is the current weather like today?'])

hf_dataset = "FredZhang7/stable-diffusion-prompts-2.47M"
dataset = load_dataset(hf_dataset)
list_all_prompts = [dataset['train'][i]['text'] for i in range(len(dataset['train']))]

# embed all prompts
fn_embeddings = 'community_prompts_embeddings.npy'
if not os.path.isfile(fn_embeddings):
    all_embeddings = np.zeros((len(list_all_prompts), 768), dtype=np.float32)
    chunk_size = 512
    nmb_chunks = int(np.ceil(len(list_all_prompts) / chunk_size))
    
    for i in tqdm(range(nmb_chunks)):
        chunk = list_all_prompts[i*chunk_size:(i+1)*chunk_size]
        emb = embedding_model.encode(chunk)
        all_embeddings[i*chunk_size:(i+1)*chunk_size] = emb
    
        # save the embeddings
        if np.mod(i, 500) == 0:
            np.save(fn_embeddings, all_embeddings)
            
    np.save(fn_embeddings, all_embeddings)
else:
    all_embeddings = np.load(fn_embeddings)
    
#%%# filter the prompts
prompt_filter = 'deep space theme'

print(f'filtering the prompts with filter: {prompt_filter}...')
filter_embeddings = embedding_model.encode([prompt_filter])
similarity = cos_sim(all_embeddings, filter_embeddings)

nmb_top = 1500
idx_top = np.argsort(similarity)[::-1][:nmb_top]

list_filtered_prompts = np.array(list_all_prompts)[idx_top].tolist()

diversifying_matrix = cos_sim_mtx(all_embeddings[idx_top, :], all_embeddings[idx_top, :])
np.fill_diagonal(diversifying_matrix, 0)

thresh_reject = 0.95
list_rejected = []

for i in tqdm(range(nmb_top)):
    idx_similar = np.where(diversifying_matrix[i,i:] > thresh_reject)[0] + i
    list_rejected.extend(idx_similar.tolist())
    
list_rejected = list(set(list_rejected))
idx_nonsimilar = np.setdiff1d(range(nmb_top), list_rejected)
list_filtered_prompts = np.array(list_filtered_prompts)[idx_nonsimilar].tolist()

# a = 'psychedelic chaos, amorphous hallucinations, eerie'
# b = 'psychedelic chaos, amorphous hallucinations, eerie'
# embeddings = embedding_model.encode([a,b])
# cos_sim(embeddings[0:1,:], embeddings[1:,:])

#%% Parameters

print('preparing the grid...')

nmb_cols = 4
nmb_rows = 4
nmb_images = nmb_cols*nmb_rows
width_diffusion = 1024
height_diffusion = 512
width_show = 512
height_show = 256
do_compile = True
num_inference_steps = 1

# 1. Inits (hf dataset, pipe, ...)
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, (height_show, width_show))
keyb = lt.KeyboardInput()


#%%

class PageGenerator:
    def __init__(self, list_filtered_prompts, pipe, gridrenderer, nmb_images, width_diffusion, height_diffusion, width_show, height_show, num_inference_steps, nmb_max_precompute=2000):
        self.list_filtered_prompts = list_filtered_prompts
        self.pipe = pipe
        self.gridrenderer = gridrenderer
        self.nmb_images = nmb_images
        self.width_diffusion = width_diffusion
        self.height_diffusion = height_diffusion
        self.width_show = width_show
        self.height_show = height_show
        self.num_inference_steps = num_inference_steps
        self.nmb_max_precompute = nmb_max_precompute
        self.list_precomputed = []
        self.prompts_precomputed = []
        self.selected_images = []
        self.selected_prompts = []
        self.is_running = True
        self.start_time = datetime.now()
        self._start_precompute_thread()

    def _precompute_images(self):
        while self.is_running:
            if len(self.list_precomputed) < self.nmb_max_precompute:
                prompt = random.choice(list_filtered_prompts)
                image = self.pipe(guidance_scale=0.0, width=self.width_diffusion, height=self.height_diffusion, prompt=prompt, num_inference_steps=self.num_inference_steps).images[0]
                image = image.resize((self.width_show, self.height_show))
                self.list_precomputed.append(np.asarray(image))
                self.prompts_precomputed.append(prompt)
            else:
                # Remove oldest precomputed images and prompts to maintain list size
                self.list_precomputed = self.list_precomputed[-self.nmb_max_precompute:]
                self.prompts_precomputed = self.prompts_precomputed[-self.nmb_max_precompute:]
            time.sleep(0.01)  # Small delay to prevent hogging CPU resources

    def _start_precompute_thread(self):
        self.precompute_thread = Thread(target=self._precompute_images)
        self.precompute_thread.daemon = True
        self.precompute_thread.start()

    def next_page(self):
        indices = random.sample(range(len(self.list_precomputed)), min(len(self.list_precomputed), self.nmb_images - 1))  # Adjust for the white tile
        img_white = Image.new('RGB', (self.width_show, self.height_show), 'white')
        img_white = lt.add_text_to_image(img_white, "click for next page", fp_font="ubuntu/Ubuntu-C", min_width=0.7)
        self.selected_images = [img_white]  # Start with a white tile
        self.selected_prompts = ["Next Page"]  # Placeholder prompt for the white tile
        self.selected_images.extend([self.list_precomputed[i] for i in indices])
        self.selected_prompts.extend([self.prompts_precomputed[i] for i in indices])
        # Remove selected images and prompts from precomputed lists
        for index in sorted(indices, reverse=True):
            del self.list_precomputed[index]
            del self.prompts_precomputed[index]
        self.gridrenderer.update(self.selected_images)

    def save_selected_prompt(self, index):
        if index < len(self.selected_prompts):
            filename = "good_prompts_" + self.start_time.strftime("%y%m%d_%H%M") + ".txt"
            with open(filename, "a") as file:
                file.write(self.selected_prompts[index] + "\n")

    def stop(self):
        self.is_running = False
        self.precompute_thread.join()

    def blend_with_green(self, m, n):
        idx = m * self.gridrenderer.nmb_cols + n
        if idx < len(self.selected_images):
            # Convert numpy array back to PIL Image for blending
            original_image = Image.fromarray(self.selected_images[idx])
            # Create a green overlay
            green_overlay = Image.new("RGBA", original_image.size, (0, 255, 0, 127))  # Semi-transparent green
            # Blend the original image with the green overlay
            blended_image = Image.alpha_composite(original_image.convert("RGBA"), green_overlay)
            # Convert blended image back to numpy array and update the selected_images list
            self.selected_images[idx] = np.asarray(blended_image.convert("RGB"))
            # Update the grid with the new blended image
            self.gridrenderer.update(self.selected_images)

# Assuming other necessary components (list_filtered_prompts, pipe, gridrenderer, etc.) are defined elsewhere in the script.
pg = PageGenerator(list_filtered_prompts, pipe, gridrenderer, nmb_images, width_diffusion, height_diffusion, width_show, height_show, num_inference_steps)
print("pre wait...")
time.sleep(5)
pg.next_page()

while True:
    m, n = pg.gridrenderer.render()
    if m != -1 and n != -1:
        idx = m * nmb_cols + n
        if idx == 0:  # Special case for the white tile
            pg.next_page()
        elif idx < len(pg.selected_prompts):
            print(f'tile index: m {m} n {n} prompt {pg.selected_prompts[idx]}')
            pg.save_selected_prompt(idx)
            pg.blend_with_green(m, n)  # Added functionality to blend selected field with green
    switch_to_next = keyb.get('n', button_mode='pressed_once')
    if switch_to_next:
        pg.next_page()