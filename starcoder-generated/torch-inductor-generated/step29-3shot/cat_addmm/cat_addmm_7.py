
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.batchnorm = nn.BatchNorm1d(num_features=4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = x.flatten(start_dim=1)
        x = self.batchnorm(x)
        x = x.unsqueeze(dim=1).unsqueeze(dim=1)
        x = torch.concat((x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
