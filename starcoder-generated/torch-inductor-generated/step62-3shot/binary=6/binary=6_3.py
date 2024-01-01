
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.value = torch.nn.Parameter(torch.Tensor([1, 2, 3, 4, 5]))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.value
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
