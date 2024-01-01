
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 5)
        self.permute = torch.transpose
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        x = self.permute(x, 0, 1)
        x = self.permute(x, 1, 2)
        x = self.stack((x, x), dim=1)
        x = x.transpose(1, 2)
        x = x.squeeze()
        return x
# Inputs to the model
x = torch.randn(2, 2, 4)
