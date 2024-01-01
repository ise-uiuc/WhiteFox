
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        x = self.stack((x, x, x), dim=1)
        x = x.view(x.shape[0], -1)
        x = x.transpose(1, 0)
        x = x.permute(1, 0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
