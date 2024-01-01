
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Softmax(dim=2)
        self.mean = torch.mean
    def forward(self, x):
        x = self.layers(x)
        x = self.mean(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(4, 5, 2)
