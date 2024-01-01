
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = x.permute(2, 1, 0) # Shape: (3, 3, 4)
        x = torch.flatten(x, start_dim=0) # Shape: (9, 4)
        return x
# Inputs to the model
x = torch.randn(4, 3)
