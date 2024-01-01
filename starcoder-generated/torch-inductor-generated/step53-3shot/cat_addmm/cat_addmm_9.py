
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.chunk(x, 5, dim=1)
        x = x[1]
        x = torch.flatten(x, start_dim=-1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
