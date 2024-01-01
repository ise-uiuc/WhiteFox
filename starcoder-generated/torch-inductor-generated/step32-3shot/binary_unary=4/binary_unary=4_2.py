
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 11)
 
    def forward(self, x1, other=None):
        if other is not None:
            v1 = self.linear(x1)
            v2 = v1 + other
            v3 = torch.nn.functional.relu(v2)
            return v3
        else:
            v1 = self.linear(x1)
            v2 = v1 + torch.Tensor([2.0, 3.0, 4.0]).to(device)
            v3 = torch.nn.functional.relu(v2)
            return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 10)
