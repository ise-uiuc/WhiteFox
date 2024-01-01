
model = torch.hub.load('pytorch/vision:v0.10.0','resnext50_32x4d', pretrained=False)
model.fc = torch.nn.Sequential(model.fc, torch.nn.ReLU())
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
