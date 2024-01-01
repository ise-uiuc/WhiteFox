
class Model(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.classifier = torch.nn.Linear(8 * 64 * 64, num_classes)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.view(1, -1)
        v3 = self.classifier(v2)
        v4 = v3 > 0
        v5 = v3 * 1e-2
        v6 = torch.where(v4, v3, v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
