
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.eye(10)
        self.linear = torch.nn.Linear(10, 1, bias=False)
        self.linear.weight = torch.nn.Parameter(w)
 
    def forward(self, x1, other=None):
        if other is None:
            raise Exception("No other tensor was passed")
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Set some_tensor as some other tensor
some_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Inputs to the model
x1 = torch.randn(1, 1, 10)
