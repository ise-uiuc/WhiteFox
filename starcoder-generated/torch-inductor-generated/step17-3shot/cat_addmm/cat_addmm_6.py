
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 10)
        self.batch_norm = nn.BatchNorm1d(5)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x, x), dim=1)
        x = self.batch_norm(x)
        return x
# Inputs to the model
x = torch.randn(2, 3)
