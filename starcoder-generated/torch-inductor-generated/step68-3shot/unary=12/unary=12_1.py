
model = torchvision.models.resnet50().eval()
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
