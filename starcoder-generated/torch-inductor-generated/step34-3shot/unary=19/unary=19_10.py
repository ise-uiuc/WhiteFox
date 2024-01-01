
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        if __use_sigmoid__:
            v2 = torch.sigmoid(v1)
        else:
            v2 = self.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
