#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import lunar_tools as lt
gpt4 = lt.GPT4(model="gpt-4-turbo-preview")    


# Open old prompts and summarize 
fn_source = 'gonsalo_prompts'
list_prompts_all = []
with open(f"{fn_source}.txt", "r", encoding="utf-8") as file: 
    list_prompts_source = file.read().split('\n')
list_prompts_source = [l for l in list_prompts_source if len(l)>0]
txt_prompts_source = "\n".join(list_prompts_source)

prompt_summarizer = f"we have the following image prompts: \n{txt_prompts_source}. \nSummarize the theme of the prompts in four words."

summary_source = gpt4.generate(prompt_summarizer)

#%% Generate new prompts\
nmb_new = 100
summary_target = "dark factory, derelict robots, techno party"
fn_target = 'prompts/robot.txt'
prompt_generator = f"You are an amazing visual artist who is able to describe his work eloquently. This is example of your work, revolving around the theme of {summary_source}: \n{txt_prompts_source}. \nNow you are making {nmb_new} new examples, however with a different theme: {summary_target}. Do not number the artworks, just make a new line for every artwork. ALWAYS make sure your artworks are DIVERSE and NON REPETETIVE, always creatively going for a new theme."
txt_prompts_target = gpt4.generate(prompt_generator)
list_prompts_target = txt_prompts_target.split('\n')
list_prompts_target = [prompt for prompt in list_prompts_target if len(prompt) > 10]
with open(f"{fn_target}", "w", encoding="utf-8") as file:
    file.write("\n".join(list_prompts_target))


