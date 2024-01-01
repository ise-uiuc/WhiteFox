
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(20, 10)
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        x = torch.reshape(x, x.shape[0], 1, 10)
        return x
# Inputs to the model
x = torch.randn(1, 1, 20, 100)
