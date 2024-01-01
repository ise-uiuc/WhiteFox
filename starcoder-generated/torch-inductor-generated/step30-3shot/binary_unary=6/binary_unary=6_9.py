
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10)
        self.bias = torch.nn.Parameter(torch.Tensor(1))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.bias
        v3 = torch.relu(v2)
        return v3

# Input to the model
x1 = torch.randn(1, 20)
