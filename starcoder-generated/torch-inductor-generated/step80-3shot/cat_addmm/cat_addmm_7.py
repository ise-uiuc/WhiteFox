
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(64, 128)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x), dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = x.flatten(1).reshape(256, 64, 128)
        return x
# Inputs to the model
x = torch.randn(2, 64)
