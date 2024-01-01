
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x).relu()
        x = torch.stack((x, x), dim=0)
        x = x.permute([1, 2, 0])
        x = torch.stack((x, x), dim=0)
        x = x.sum(dim=1)
        x = x.view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
