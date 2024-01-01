
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x.repeat(1, x.numel()), x.repeat(2, 1), x, x.view(4)), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
