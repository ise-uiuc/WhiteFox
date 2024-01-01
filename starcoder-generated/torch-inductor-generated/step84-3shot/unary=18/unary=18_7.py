
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.alexnet(pretrained=True)
    def forward(self, x1):
        v1 = self.model.avgpool(x1)
        v2 = self.model.relu(v1)
        v3 = self.model.reshape(v2, -1)
        v4 = self.model.classifier[1](v3)
        v5 = self.model.classifier[5](v4)
        return v5
# Inputs to model
x1 = torch.randn(1, 3, 224, 224)
