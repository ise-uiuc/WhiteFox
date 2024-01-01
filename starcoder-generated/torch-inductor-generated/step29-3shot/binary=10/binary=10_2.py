
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(14, 8)
        self.linear2 = torch.nn.Linear(8, 10)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = v2 + other_tensor
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14)
