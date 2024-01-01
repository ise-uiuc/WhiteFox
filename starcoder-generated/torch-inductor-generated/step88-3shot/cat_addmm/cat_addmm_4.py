
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 3)
        self.layers_2 = nn.Linear(3, 3)
    def forward(self, x_1, x_2):
        x_1 = self.layers_1(x_1)
        x_1 = torch.stack((x_1, x_1), dim=1).flatten(1)
        x_2 = self.layers_2(x_2)
        x_2 = torch.stack((x_2, x_2), dim=1).flatten(1)
        x = torch.cat((x_1, x_2), dim=1)
        return x
# Inputs to the model
x_1 = torch.randn(2, 2)
x_2 = torch.randn(2, 3)
