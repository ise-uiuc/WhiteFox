
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 5)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        a = x1.permute(0,2,1)
        b = torch.nn.functional.linear(a, self.linear.weight, self.linear.bias)
        c = self.softmax(b)
        d = c.permute(0,2,1)
        return d
# Inputs to the model
x1 = torch.randn(1, 5, 16)
