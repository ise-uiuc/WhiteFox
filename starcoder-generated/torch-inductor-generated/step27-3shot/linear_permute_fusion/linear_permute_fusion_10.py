
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.linear.forward(x1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 2, 2)
