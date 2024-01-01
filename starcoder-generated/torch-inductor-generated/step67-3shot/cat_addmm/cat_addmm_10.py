
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(24, 16)
        self.second_layer = nn.Linear(16, 24)
    def forward(self, x):
        x = self.first_layer(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.norm(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 24)
