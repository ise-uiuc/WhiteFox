
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 3 * 1, 1, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = tensor([[[[1, 10, 11], [12, 13, 14], [15, 16, 17]]], [[[2, 19, 20], [21, 22, 23], [24, 25, 26]]], [[[3, 28, 29], [30, 31, 32], [33, 34, 35]]]])
