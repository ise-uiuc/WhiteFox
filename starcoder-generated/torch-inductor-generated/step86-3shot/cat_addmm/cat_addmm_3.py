
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x, x), dim=2)
        x = x.transpose(dim0=1, dim1=2)
        return x
# Inputs to the model
x = torch.randn(1, 2)
