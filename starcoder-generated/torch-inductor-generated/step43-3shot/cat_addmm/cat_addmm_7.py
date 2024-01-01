
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.ones(1, 6)
        (x_1, x_2) = torch.chunk(x, 2, dim=1)
        x_1 = torch.clone(x_1)
        x_2 = torch.clone(x_2)
        result = torch.stack((x_1, x_2), dim=1)
        return result
# Inputs to the model
x = torch.randn(2, 2)
