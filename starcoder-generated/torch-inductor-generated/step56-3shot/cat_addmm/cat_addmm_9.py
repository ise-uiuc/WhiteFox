
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(5, 8)
        self.layers_2 = nn.Linear(8, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = F.relu(x)
        x = torch.stack((x, x), dim=1)
        x = x[:, 1:3]
        x = x.flatten(start_dim=1)
        x = F.relu(x)
        x = self.layers_2(x)
        y = x.transpose(1, 0)
        return x
# Inputs to the model
x = torch.randn(2, 5)
