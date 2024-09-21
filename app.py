import PIL.Image
import gradio as gr
import torch
from torch import nn
import timm
from PIL import Image
from classes import classes
import torchvision.transforms.functional as TF
from pathlib import Path
import os 

device = 'cpu'
def get_model():
    model = timm.create_model('levit_128s.fb_dist_in1k', pretrained=True)
    model.eval()
    model.fc = nn.Linear(2048, 101)
    model.head.linear = nn.Linear(384, 101)
    model.head_dist.linear = nn.Linear(384, 101)
    return model.to(device)

def image_classifier(inp):
    img = Image.fromarray(inp)
    inp = tfs(img).unsqueeze(0)
    out = model(inp)
    preds = out.softmax(1)
    top_5 = torch.argsort(preds, dim=1)[:5]
    return {classes[i.item()]: preds[0][i.item()] for i in top_5[0]}

model = get_model()
model.load_state_dict(torch.load('model.pt', map_location=torch.device(device), weights_only=True))
data_config = timm.data.resolve_model_data_config(model)
tfs = timm.data.create_transform(**data_config, is_training=False)

nms = os.listdir('./examples')
examples = []
for nm in nms:
    examples.append(Image.open(Path('examples') / nm))

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label", examples=examples)
demo.launch()