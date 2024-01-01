
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8, bias=False)
        self.linear.weight = torch.nn.Parameter(torch.zeros((8, 4)))
        for i in range(4):
            self.linear.weight[i, i] = 1
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + some_tensor
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
