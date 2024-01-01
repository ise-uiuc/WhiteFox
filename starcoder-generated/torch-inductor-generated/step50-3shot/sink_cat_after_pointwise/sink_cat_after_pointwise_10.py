
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def foo(self, x):
        shape = x.shape
        y = x.reshape(1, -1)
        return y.sum() if shape[0] == 1 else y.view(-1)
    def forward(self, x, y):
        x = self.foo(x)
        y = self.foo(y)
        return (x + y)
# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(2, 3)
