
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v3 = F.relu(v1 + other=a)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
a = torch.zeros(1, 8)
# For different executions, please generate different tensors for `a`.
