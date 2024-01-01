
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.relu
    def forward(self, x):
        x1 = torch.cat([x, x, x, x], dim=0)
        return self.op(x1)
# Inputs to the model
x = torch.randn(2, 3, 4)
