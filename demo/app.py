import torch
import torchvision
import gradio as gr
from time import time

with open('classes.txt', 'r') as file:
    CLASS_NAMES = [name.strip().split('-', 1)[1] for name in file.readlines()]
    NUM_CLASSES = len(CLASS_NAMES)

transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

examples = [
    'examples/saluki.jpg',
    'examples/bull.jpg',
    'examples/haski.jpg'
]

model = torch.load('model.pth', map_location=torch.device('cpu'))

def prob_and_pred(img):
    model.eval()
    with torch.inference_mode():
        start_time = time()
        transformed_img = transformer(img.convert('RGB')).unsqueeze(0)
        logits = model(transformed_img)
        probs = torch.softmax(logits, dim=1)
        end_time = time()
        
        prediction_dict = {CLASS_NAMES[index]: probs[0][index].item() for index in range(NUM_CLASSES)}
        prediction_time = round(end_time - start_time, 4)
        
        return prediction_dict, prediction_time

demo = gr.Interface(
    fn=prob_and_pred,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Label(num_top_classes=3, label='Prediction Probs'), gr.Number(label='Prediction Time')],
    examples=examples,
    title='Dog Vision',
    description='Upload your dog photo to detect its breed with over 90% accuracy!'
)

demo.launch()