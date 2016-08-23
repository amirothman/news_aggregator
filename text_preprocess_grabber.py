from pathlib import Path
import json
import numpy as np
import re

def convert_emotional_values(doc):
    if 'amused' in doc.keys():
        happy = float(doc["happy"])/100+float(doc["amused"])/100+float(doc["inspired"])/100
        angry = float(doc["angry"])/100+float(doc["annoyed"])/100
        sad = float(doc["sad"])/100
        return [happy,angry,sad]

    else:
        if [doc["happy"],doc["angry"],doc["sad"]] == [0,0,0]:
            return [0.0,0.0,0.0]
        else:
            total = float(doc["happy"]) + float(doc["sad"]) + float(doc["angry"])
            return [float(doc["happy"])/total,float(doc["angry"])/total,float(doc["sad"])/total]

def lower_remove_new_line(text):
    t = re.sub(r"\n"," ", text)
    return t.lower()

collection_folder_path = "/home/amir/makmal/ular_makan_surat_khabar/the_star"
p = Path(collection_folder_path)
labels = ["__label__happy","__label__angry","__label__sad"]
with open("fast_text_input.txt","w") as fast_text_input:
    for j in p.glob("**/*.json"):
        dictionary = json.loads(j.read_text())
        emotional_values = convert_emotional_values(dictionary)
        label = labels[np.argmax(emotional_values)]
        text = lower_remove_new_line(dictionary["content"])
        fast_text_input.write(label)
        fast_text_input.write(" ")
        fast_text_input.write(text)
        fast_text_input.write("\n")
