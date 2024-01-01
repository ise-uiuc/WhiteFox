
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 1)
    def forward(self, x):
        x = self.layers(x)
        return torch.squeeze(x)
# Inputs to the model
x = torch.randn(3, 2, 1)
