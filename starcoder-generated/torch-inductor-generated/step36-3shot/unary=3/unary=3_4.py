
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
    def forward(self, x0):
        v1 = self.relu1(x0)
        return v1
# Inputs to the model
x0 = torch.randn(1)
