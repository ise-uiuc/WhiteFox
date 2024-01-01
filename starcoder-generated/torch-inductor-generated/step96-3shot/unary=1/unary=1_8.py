
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(12, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 32)
        self.linear4 = torch.nn.Linear(32, 16)
        self.linear5 = torch.nn.Linear(16, 8)
        self.linear6 = torch.nn.Linear(8, 4)
 
    def forward(self, x2):
        v1 = self.linear1(x2)
        v2 = v1 * 0.5
        w1 = self.linear6(x2)
        w2 = w1 * 0.5
        v3 = v1 + (v1 * v1 * v1) * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        w3 = w2 + (w2 * w2 * w2) * 0.044715
        w4 = w3 * 0.7978845608028654
        w5 = torch.tanh(w4)
        w6 = w5 + 1
        w7 = w2 * w6
        z1 = v7 + w7
        return z1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 12)
