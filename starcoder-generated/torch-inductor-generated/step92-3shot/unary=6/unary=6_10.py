
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6(inplace=False)
    def forward(self, x1):
        v1 = self.relu6(x1)
        return
# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
