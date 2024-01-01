
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Flatten()
    def forward(self, x):
        return self.op(x)
# Inputs to the model
x = torch.randn(1, 4, 5, 5)
