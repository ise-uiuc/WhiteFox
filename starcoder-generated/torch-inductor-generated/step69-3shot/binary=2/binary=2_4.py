
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.relu1(x1)
        v2 = v1 - 10
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
