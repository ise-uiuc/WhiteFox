
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 8)
    def forward(self, x):
        x = self.layers(x)
        x = x + 0.01
        x = torch.stack((x, x), dim=1)
        x = x[[0,1], [0,1,1,0,1,1,0,1]]
        x = torch.nn.Flatten(start_dim=0, end_dim=1)(x)
        return x
# Inputs to the model
x = torch.randn(2, 8)
