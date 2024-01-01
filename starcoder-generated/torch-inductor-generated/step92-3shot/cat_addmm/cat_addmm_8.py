
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(1, 128)
        self.layers_2 = nn.Linear(128, 256)
    def forward(self, x):
        x = self.layers_1(x)
        x = torch.stack((x, x), dim=-1)
        x = torch.max(x, dim=0).values
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 1)
