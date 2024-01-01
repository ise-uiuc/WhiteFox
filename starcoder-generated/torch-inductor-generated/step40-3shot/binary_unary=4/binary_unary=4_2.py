
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is not None:
            v1 += other
        v2 = torch.relu(v1)
        return v2

# Inputs to the model
x1 = torch.randn(1, 10)
other = torch.randn(5)
