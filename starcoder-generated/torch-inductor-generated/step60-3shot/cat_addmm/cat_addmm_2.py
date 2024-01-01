
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(2, 4, 1)
        x = torch.cat((x, x), dim=1)
        x = torch.transpose(x, 1, 2).flatten(2)
        return x
# Inputs to the model
x = torch.randn(2, 2)
