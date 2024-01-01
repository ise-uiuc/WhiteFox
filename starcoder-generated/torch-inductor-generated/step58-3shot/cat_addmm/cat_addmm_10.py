
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.reshape(x, (-1, x.shape[2], 2))
        x = torch.stack((x, x))
        return x
# Inputs to the model
x = torch.randn(2, 2, 2, 4)
