
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.rand(24, 24)
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.nn.functional.linear(x1, self.weight)
        v2 = torch.nn.functional.linear(x2, self.weight)
        v3 = x3.transpose(-2, -1)
        v4 = torch.matmul(v1, v2)
        v5 = v4.div(20.0)
        v6 = torch.softmax(v5, -1)
        v7 = torch.nn.functional.dropout(v6, 0.3, False)
        v8 = v7.matmul(x4)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 24)
x2 = torch.randn(16, 24)
x3 = torch.randn(16, 32, 24)
x4 = torch.randn(16, 32, 5)
