
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 3)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        input = torch.stack((x, x, x), dim=0)
        x = self.cat((input, input, input), dim=1)
        return x
# Inputs to the model
x = torch.randn(4, 16)
