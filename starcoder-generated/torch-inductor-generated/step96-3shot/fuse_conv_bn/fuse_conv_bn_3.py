
# TODO add code after "model = torchvision.models.resnet18()"
torch.manual_seed(0)
model = torchvision.models.resnet18()
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

# Inputs to the model
x3 = torch.randn(2, 3, 224, 224)
