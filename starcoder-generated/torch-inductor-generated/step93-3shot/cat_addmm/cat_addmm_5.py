
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x_1 = self.layers_1(x)
        x_2 = self.layers_1(x)
        x = torch.cat((x, x_1, x_2), dim=1)
        x = x.reshape(-1, 2, 2)
        x = torch.cat((x, x), dim=1)
        x = x.flatten(end_dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
