
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(20, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.zeros(x.shape[0], 200, dtype=torch.float)
        x[0, 3:30] = x[0, 10:110]
        x[1, 3:30] = x[1, 10:110]
        x = x.view(x.shape[0], 5, 10, 2)
        x = torch.sum(x, dim=3)
        x = torch.exp(x)
        x = 0.01 * x
        x = torch.softmax(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 20)
