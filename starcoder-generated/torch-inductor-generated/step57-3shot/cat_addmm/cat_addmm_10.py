
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 1)
    def forward(self, x):
        x = x.repeat(1,10)
        x = self.layers(x)
        x = x[0, 0]
        return x
# Inputs to the model
x = torch.arange(24)
x = x.reshape(2, 3, 4)
