
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = torch.nn.Linear(6,4)
        self.layer3 = torch.nn.Linear(4, 1)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Linear(1, 3)
        self.layer6 = torch.nn.Sigmoid()
        self.layer7 = torch.nn.Linear(3, 6)
    def forward(self, x1):
        v1 = self.layer2(x1)
        v2 = self.layer3(v1)
        v3 = self.layer4(v2)
        v4 = self.layer5(v3)
        v5 = self.layer6(v4)
        v6 = self.layer7(v5)
        return v6
# Inputs to the model
x1 = torch.randn(2, 6)
