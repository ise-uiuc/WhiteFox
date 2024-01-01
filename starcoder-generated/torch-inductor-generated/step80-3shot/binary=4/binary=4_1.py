
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10, bias=True)
        self.bias = torch.nn.Parameter(torch.Tensor(10))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.bias
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7, 28 * 28)

# Setting the "other" tensor (the tensor to add together with the output of the model)
other = torch.zeros(10)
