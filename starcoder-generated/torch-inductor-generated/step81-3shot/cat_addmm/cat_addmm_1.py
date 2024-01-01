
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=0)
        return x # This is wrong
# Inputs to the model
x = torch.randn(2, 2)
