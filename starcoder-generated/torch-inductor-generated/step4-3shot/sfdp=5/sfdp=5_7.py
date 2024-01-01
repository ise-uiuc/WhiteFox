
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(768, 3072)
        self.linear2 = torch.nn.Linear(3072, 768)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.tanh(v1)
        v3 = self.linear2(v2)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768)
