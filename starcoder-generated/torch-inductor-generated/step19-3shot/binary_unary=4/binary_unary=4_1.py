
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False)
 
    def forward(self, x1, other=None):
        if other is None:
            other = torch.randn(3, 4)
        elif other.dim()!= 2:
            raise ValueError(f'The shape of "other" is expected to be [3, 4], but got {list(other.shape)}')
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 288, 352)
