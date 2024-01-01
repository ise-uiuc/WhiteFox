
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(5):
            setattr(self, 'layer' + str(i + 1), torch.nn.Linear(1, 1, bias=False))
    def forward(self, x1, x2):
        v1 = self.layer1(x1)
        v2 = self.layer2(x1)
        v3 = self.layer3(x1)
        v4 = self.layer4(x1)
        return torch.cat([v1, v2, v3, v4], 1)
# Inputs to the model
x1 = torch.randn(4, 1)
x2 = torch.randn(4, 1)
