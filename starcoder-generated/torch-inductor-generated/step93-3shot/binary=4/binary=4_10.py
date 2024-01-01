
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        print(self.linear.bias)
        v2 = v1 + self.linear.bias
        return v2

m = Model()
x1 = torch.randn(1, 10)
