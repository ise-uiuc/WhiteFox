
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 32)
    def forward(self, x):
        x = self.layers(x)
        x_1 = x[0]
        x_2 = x[1]
        x_12 = torch.cat((x_1, x_2), dim=-1)
        x_21 = torch.cat((x_2, x_1), dim=-1)
        x = torch.cat((x_12, x_21), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 16)
