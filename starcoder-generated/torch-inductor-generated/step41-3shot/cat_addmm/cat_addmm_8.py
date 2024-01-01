
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers(x)
        x = x.permute(1, 0)
        x = torch.cat((x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 3)
