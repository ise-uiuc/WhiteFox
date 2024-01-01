
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8 * 8 * 3, bias=False)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 - 8
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
