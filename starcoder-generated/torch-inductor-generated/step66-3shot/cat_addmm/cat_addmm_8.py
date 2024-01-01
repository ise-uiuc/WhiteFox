
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x))
        x = x.transpose(2, 1)
        x = x.reshape(1, 8)
        return x
# Inputs to the model
x = torch.randn(4, 1)
