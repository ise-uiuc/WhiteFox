
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg11(pretrained=True)
    def forward(self, x):
        x = self.vgg.features(x)
        y = torch.tanh(x)
        return y
# Inputs to the model
tensor = torch.randn(1, 3, 224, 224)
