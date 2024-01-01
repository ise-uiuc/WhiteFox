
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
    def forward(self, x):
        x = self.linear1(x)
        y = torch.rand_like(x)
        y += torch.rand_like(x)
        return y
# Inputs to the model
x = torch.randn(1, 10)
