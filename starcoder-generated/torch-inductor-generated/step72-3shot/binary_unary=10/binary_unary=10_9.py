
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(256, 1000)

    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = v1.view(v1.size(0), -1)
        v3 = self.classifier(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000, 64, 64)
