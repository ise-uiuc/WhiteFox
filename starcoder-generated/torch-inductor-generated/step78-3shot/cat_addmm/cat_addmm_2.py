
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=0)
        x = torch.flatten(x)
        x = x.unsqueeze(dim=0)
        x = torch.add(x, x)
        x = x.squeeze(dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
