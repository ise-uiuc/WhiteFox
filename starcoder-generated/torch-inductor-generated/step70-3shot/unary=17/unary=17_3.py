
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self, x1):
        v1 = torch.flatten(x1, 1)
        v2 = self.softmax(v1)
        return torch.reshape(v2, (1, 3, 1, 1))
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
