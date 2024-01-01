
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=True)
        torch.manual_seed(0)
        self.linear.weight = torch.nn.Parameter(torch.randn(self.linear.weight.size()))

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.relu(v1 - torch.tensor([1.0],dtype=torch.float32))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, dtype=torch.float32)
