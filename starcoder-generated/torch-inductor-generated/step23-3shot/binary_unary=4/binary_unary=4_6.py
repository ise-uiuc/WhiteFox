
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.linear.weight = weight
        self.linear.bias = bias
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3
 
# Initializing the model
weight = torch.tensor([[1.0, 2.0]]).float()
bias = torch.tensor([3.0]).float()
m = Model(weight, bias)
 
# Inputs to the model
x1 = torch.tensor([[1.0, 2.0]]).float()
x2 = torch.tensor([3.0]).float()
