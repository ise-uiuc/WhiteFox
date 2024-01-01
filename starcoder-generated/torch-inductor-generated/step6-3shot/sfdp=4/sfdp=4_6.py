
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v_linear = torch.nn.Linear(8, 64)
        self.p_linear = torch.nn.Linear(16, 64)
 
    def forward(self, x1, x2, x3):
        v1 = self.v_linear(x1)
        v2 = self.p_linear(x2)
        v3 = v1 @ v2.T / math.sqrt(v1.size(-1))
        v4 = v3 + x3
        v5 = torch.softmax(v4, dim=-1)
        v = v5 @ v2
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 16)
x3 = torch.FloatTensor([[[[2.0, -1.0, -1.0, 2.0, 1.0, 4.0, -4.0, -4.0]]]])
