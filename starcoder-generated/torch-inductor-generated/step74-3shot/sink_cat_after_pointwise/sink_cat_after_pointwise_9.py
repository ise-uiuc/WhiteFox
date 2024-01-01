
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def foo(x):
            return x.transpose(0, 1)
        self.foo = foo
    def forward(self, x):
        y = torch.cat((x, x, x), dim=0)
        y = self.foo(y)

        return y.view(y.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3)
