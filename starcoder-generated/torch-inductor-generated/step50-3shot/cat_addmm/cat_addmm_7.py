
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        return x.flatten(start_dim=0)
# Inputs to the model
x = torch.randn(2, 2)
