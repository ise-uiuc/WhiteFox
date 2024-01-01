
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6, bias=True)
        self.other = torch.Tensor([1.5])
 
    def forward(self, x1):
        v0 = x1.flatten(1)
        v1 = self.linear(v0)
        v2 = v1 - self.other
        v3 = relu(v2)
 
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
