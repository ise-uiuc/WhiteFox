
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
 
    def forward(self, x1, t1):
        v1 = self.linear(x1)
        v2 = v1 + t1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
t1 = torch.Tensor([0.1])
