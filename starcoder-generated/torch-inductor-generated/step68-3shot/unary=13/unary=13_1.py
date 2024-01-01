
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1024, bias=False)
        self.linear.weight.data = torch.rand_like(self.linear.weight.data)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = torch.sigmoid(x1)
        x3 = x1 * x2
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
