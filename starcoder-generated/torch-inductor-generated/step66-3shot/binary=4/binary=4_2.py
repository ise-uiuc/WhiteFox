
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        W = torch.randn(1, 3, 64, 64)
        self.linear = torch.nn.Linear(8192, 10, bias=False)
        self.linear.weight[:] = W.reshape(W.shape[0], -1)
 
    def forward(self, x1, other=0):
        v1 = self.linear(x1.reshape(x1.shape[0], -1))
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
