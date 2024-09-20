import gradio as gr
import torch
from torch import nn
import timm
import PIL
from classes import classes

device = 'cpu'
def get_model():
    model = timm.create_model('levit_128s.fb_dist_in1k', pretrained=True)
    model.eval()
    model.fc = nn.Linear(2048, 101)
    return model.to(device)

model = get_model()
data_config = timm.data.resolve_model_data_config(model)
tfs = timm.data.create_transform(**data_config, is_training=False)
model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))

def image_classifier(inp):
    inp = tfs(torch.tensor(inp).permute(2,0,1).unsqueeze(0).float())
    out = model(inp)
    preds = out.softmax(1)
    top_5 = torch.argsort(preds, dim=1)[:5]
    import pdb; pdb.set_trace()
    return {classes[i.item()]: preds[i.item()] for i in top_5}

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()