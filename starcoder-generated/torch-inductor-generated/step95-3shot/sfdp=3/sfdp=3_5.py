
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 1.7574051560137903
        v3 = torch.nn.functional.tanh(v2)
        v4 = torch.mul(v3, v3)
        v5 = 0.205
        v6 = v4.mul(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 4)
x2 = torch.randn(1, 4, 16)
