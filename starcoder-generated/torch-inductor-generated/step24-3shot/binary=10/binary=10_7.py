 1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=16, out_features=8, bias=True)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + x1.sum([1, 2, 3])
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16, 1, 1)
