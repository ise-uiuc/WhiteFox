
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.empty(1, 5, dtype=v1.dtype, layout=v1.layout, device=v1.device)
        v2.fill_(0.1)
        v3 = v1 - v2
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, dtype=torch.float32)
