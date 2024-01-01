
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = torch.cat
    def forward(self, x):
        x = x.flatten(1)
        return self.cat((x, x.t()), dim=1)
# Inputs to the model
x = torch.randn(3, 5)
