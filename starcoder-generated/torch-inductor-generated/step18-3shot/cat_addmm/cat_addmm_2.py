
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.mul(x, x)
        x = torch.cat((x, x, x, x), dim=-1)
        x = F.relu(x, inplace=False)
        return x
# Inputs to the model
x = torch.randn(2, 1)
