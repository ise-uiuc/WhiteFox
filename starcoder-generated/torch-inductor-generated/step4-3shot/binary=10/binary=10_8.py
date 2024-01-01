
class Model(torch.nn.Module):
    def __init__(self, other_tensor_a, other_tensor_b):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other_tensor_a
        v3 = v1 + other_tensor_b
        return v2, v3

# Initializing the model
other_tensor_a = torch.randn(10, 5)
other_tensor_b = torch.randn(10, 5)
m = Model(other_tensor_a, other_tensor_b)

# Inputs to the model
x1 = torch.randn(1, 5)
__input__ = x1
