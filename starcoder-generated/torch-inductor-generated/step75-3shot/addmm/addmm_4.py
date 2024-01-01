
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear_ = torch.nn.Linear(10, 10, bias=False)
        torch.nn.init.eye_(self.linear_.weight)

    def forward(self, x, y):
        return self.linear_(x).matmul(y)
# Inputs to the model
x = torch.randn(7, 10)
y = torch.randn(10, 7)
