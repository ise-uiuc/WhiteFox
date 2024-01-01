
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = torch.zeros(2, 2, dtype=torch.float) # Add an arbitrary number of extra "ones" tensors to trigger the bug
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
