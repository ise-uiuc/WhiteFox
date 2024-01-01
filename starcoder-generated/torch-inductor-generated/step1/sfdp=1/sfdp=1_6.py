
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(16, 32)
        self.m2 = torch.nn.Linear(32, 64)
 
    def forward(self, m1_x, m2_x):
        v1 = self.m1(m1_x)
        v2 = self.m2(v1)
        v3 = torch.matmul(m2_x, v2.t())
        v4 = v3 / (math.sqrt(v2.shape[-1]) or 1)
        v5 = torch.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.1)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
m1_x = torch.randn(1, 16)
m2_x = torch.randn(1, 64, 32)
