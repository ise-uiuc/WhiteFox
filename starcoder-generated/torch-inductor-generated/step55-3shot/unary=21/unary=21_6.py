
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.fc = torch.nn.Linear(in_features=1000, out_features=2)
    def forward(self, x):
        v1 = self.vgg16(x)
        v2 = torch.tanh(v1)
        v3 = torch.flatten(v2, start_dim=1)
        v4 = self.fc(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
