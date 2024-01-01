
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, other=None):
        if other is None:
            other = torch.rand_like(x1, dtype=x1.dtype)
        if not torch.is_tensor(other):
            raise TypeError("other must be a Torch tensor; " "but other.dtype == %s" % other.dtype)
 
        v1 = self.linear(x1)
        v2 = v1 + other
 
        return v2

# Inputs to the model
x1 = torch.randn(1, 64, 8, 8)
