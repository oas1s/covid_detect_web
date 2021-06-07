import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
from PIL import Image
import torch.nn.functional as F

model = models.densenet169(pretrained=True).cuda()
pretrained_net = torch.load('Self-Trans.pt')
model.load_state_dict(pretrained_net)
model.eval()
device = 'cuda'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transformer = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


def takeProbability(tensor):
    if tensor[0].item() > tensor[1].item():
        return round(tensor[0].item()*100, 2)
    else:
        return round(tensor[1].item()*100, 2)


def predict(file):
    image = Image.open(file).convert('RGB')
    tensor = val_transformer(image).cuda()
    # print(tensor.shape)
    tensor = torch.unsqueeze(tensor, 0)
    # print(tensor.shape)
    output = model(tensor)
    print(output[0, :2])
    pred = model(tensor).argmax(dim=1, keepdim=True)
    prob = takeProbability(F.softmax(output[0, :2]))
    prediction = [pred,prob]
    return prediction
