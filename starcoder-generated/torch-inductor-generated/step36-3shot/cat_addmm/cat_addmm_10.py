
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4),
                                    nn.Softmax(dim=1))
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
