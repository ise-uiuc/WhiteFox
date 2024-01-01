
class Model(torch.nn.Module):
    def __init__(self, size, bias=True, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(3, size, None, bias)
        self.dtype = dtype
 
    def forward(self, x1):
        x1 = torch.tensor(x1, dtype=self.dtype)
        x2 = self.linear.weight
        x3 = torch.matmul(x1, x2)
 
        if self.linear.bias is not None:
            x3 += self.linear.bias
        x4 = torch.relu(x3)
        x5 = x4 - x4 # Remove the useless computation to make the generated example simple
        return torch.sum(x3)

# Initializing the model
m = Model(8)

# Inputs to the model
x1 = [[17, 2, 4], [1, 1, 1], [5, 12, 0]]
