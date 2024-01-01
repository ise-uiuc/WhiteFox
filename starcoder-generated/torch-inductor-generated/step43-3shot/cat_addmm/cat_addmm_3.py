
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x_1 = x.unsqueeze(1)
        x_2 = x.unsqueeze(0)
        x = x_1 + x_2
        (x_1, x_2) = torch.chunk(x, 2, dim=0)
        (x_1, _) = torch.chunk(x_1, 2, 1)
        (x_2, _) = torch.chunk(x_2, 2, 1)
        x = torch.cat((x_1, x_2), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2).float()
