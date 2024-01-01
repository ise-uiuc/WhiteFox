
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.detach()
        x = torch.stack((x, x, x), dim=1)
        x = torch.flatten(x, 1)
        x = torch.cat((x.unsqueeze(0), x.unsqueeze(0)), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
