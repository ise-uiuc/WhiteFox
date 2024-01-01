
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(12, 7)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x), dim=0)
        x = x.view(-1, 3, 4)
        x = x.permute(0, 2, 1)
        x = x.flatten(0, 1)
        return x
# Inputs to the model
x = torch.randn(3, 5)
