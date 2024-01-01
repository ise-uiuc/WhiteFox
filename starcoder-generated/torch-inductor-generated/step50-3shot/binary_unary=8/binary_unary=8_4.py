
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x1):
        v1 = torch.nn.Flatten()
        v2 = self.softmax(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 16, 2, 2)
