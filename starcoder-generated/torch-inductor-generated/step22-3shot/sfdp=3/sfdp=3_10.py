
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(64, 32)
        self.linear_k = torch.nn.Linear(64, 32)
        self.scale_factor = math.sqrt(32)
 
    def forward(self, x1, x2):
        v1 = self.linear_q(x1)
        v2 = self.linear_k(x2)
        v3 = torch.matmul(v1, v2)
        v4 = v3.mul(self.scale_factor)
        v5 = v4.softmax(dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.5)
        v7 = torch.matmul(v6, x2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 64)
x2 = torch.randn(2, 64)
