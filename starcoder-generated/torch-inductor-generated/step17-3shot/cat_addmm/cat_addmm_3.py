
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 8)
    def forward(self, x):
        x = self.layers(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, x, x, x), dim=-1)
        return x
# Inputs to the model
x = torch.randn(7, 16)
