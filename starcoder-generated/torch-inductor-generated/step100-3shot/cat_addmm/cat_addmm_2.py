
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 2)
        torch.Tensor.add_ = torch.Tensor.add
    def forward(self, x):
        y = self.layers(x)
        z = x + 5
        w = self.layers(y)
        res = z + w
        return res

# Inputs to the model
x = torch.randn(2, 1)
