import torch
import numpy as np
import random
import lunar_tools as lt
import time

@staticmethod
@torch.no_grad()

#%% Aux Func and classes
class PromptManager:
    def __init__(self, use_community_prompts, prompts=None):
        self.use_community_prompts = use_community_prompts
        self.prompts = prompts
        self.current_prompt_index = 0  # To track the current index if prompts is a list
        self.hf_dataset = "FredZhang7/stable-diffusion-prompts-2.47M"
        self.local_prompts_path = "../psychoactive_surface/good_prompts.txt"
        self.fp_save = "good_prompts_harvested.txt"
        if self.use_community_prompts:
            self.dataset = load_dataset(self.hf_dataset)
        else:
            self.list_prompts_all = self.load_local_prompts()

    def load_local_prompts(self):
        with open(self.local_prompts_path, "r", encoding="utf-8") as file:
            return file.read().split('\n')

    def get_new_prompt(self):
        if isinstance(self.prompts, str):  # Fixed prompt provided as a string
            return self.prompts
        elif isinstance(self.prompts, list):  # List of prompts provided
            prompt = self.prompts[self.current_prompt_index]
            self.current_prompt_index = (self.current_prompt_index + 1) % len(self.prompts)  # Loop through the list
            return prompt
        
        else:
            # Fallback to random prompt selection if no fixed or list of prompts provided
            if self.use_community_prompts:
                return random.choice(self.dataset['train'])['text']
            else:
                return random.choice(self.list_prompts_all)

    def save_harvested_prompt(self, prompt):
        with open(self.fp_save, "a", encoding="utf-8") as file:
            file.write(prompt + "\n")
