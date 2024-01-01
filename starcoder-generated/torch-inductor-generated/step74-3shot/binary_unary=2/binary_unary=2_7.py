
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1000, 1000)
        self.layer2 = torch.nn.Linear(1000, 1000)
        self.layer3 = torch.nn.Linear(1000, 1)
    def forward(self, x1):
        v1 = self.layer2(x1)
        v2 = v1 * 0.5
        v3 = self.layer1(v2)
        v4 = v3 * 0.5
        v5 = self.layer3(x1)
        v6 = v5 * 0.25
        v7 = v3 * v5
        return v7
# Inputs to the model
x1 = torch.randn(2, 1000)
