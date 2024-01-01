
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        w1 = torch.ones(5, 5)
        self.linear = torch.nn.Linear(20, 20, dtype=torch.float, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.mm(w1, w1.t()))

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 20)
