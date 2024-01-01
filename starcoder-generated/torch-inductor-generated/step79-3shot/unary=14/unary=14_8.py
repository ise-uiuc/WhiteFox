
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.relu1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 256, 56, 56)
