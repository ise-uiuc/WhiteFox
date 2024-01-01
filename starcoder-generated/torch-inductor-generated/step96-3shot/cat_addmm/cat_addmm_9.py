
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(7, 6)
        self.layers_2 = nn.Linear(6, 5)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.layers_2(x)
        if bool(random.getrandbits(1)):
            x = torch.transpose(x, 0, 1)
        x = torch.cat((x, x, x), dim=0)
        x = x.T
        return x
# Inputs to the model
x = torch.randn(3, 7)
