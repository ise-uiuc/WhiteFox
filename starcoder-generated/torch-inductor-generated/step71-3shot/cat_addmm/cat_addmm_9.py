
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(2, 2, 2)
        self.layer2 = nn.Conv2d(2, 1, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = x.flatten(2).flatten(1)
        x = self.layer2(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2, 2)
