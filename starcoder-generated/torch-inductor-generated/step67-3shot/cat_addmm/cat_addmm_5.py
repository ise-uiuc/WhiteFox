
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1024, 1024)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(1024, 1024)
