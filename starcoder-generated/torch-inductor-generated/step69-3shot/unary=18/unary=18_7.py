
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torchvision.models.squeezenet1_1(pretrained=True)
        del self.conv1.classifier
        del self.conv1.avgpool
    def forward(self, x1):
        v1 = self.conv1.features(x1)
        v2 = self.conv1.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
