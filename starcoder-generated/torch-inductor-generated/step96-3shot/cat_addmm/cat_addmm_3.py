
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(7, 6)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = torch.cat((x, x), dim=3)
        x = x.view(2, 7, 2)
        x = x.permute(1, 0, 2)
        x = torch.cat((x, x), dim=-1)
        x = x.flatten(end_dim=2)
        return x
# Inputs to the model
x = torch.randn(7, 7)
