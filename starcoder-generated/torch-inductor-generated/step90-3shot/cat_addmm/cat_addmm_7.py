
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.view((x.shape[-1], 1))
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.cat((x, x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 4)
