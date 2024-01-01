
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = (nn.Linear(2, 2) + 1) * 2
        self.layer2 = nn.Conv2d(1, 1, 3, groups=1) - 1
        self.layer3 = nn.Sigmoid()
        self.layer4 = nn.LayerNorm([1,2,3])
        self.layer5 = ((nn.LayerNorm([1,2,3]) * nn.Sigmoid())
                       - (nn.Conv2d(1, 1, 3) + 2))
    def forward(self, x):
        x = self.layer1 
        x = self.layer2
        x = self.layer3(self.layer4[self.layer5](x))
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
