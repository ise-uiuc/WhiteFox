
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.q = torch.nn.Linear(64, 64)
        self.k = torch.nn.Linear(64, 64)
        self.v = torch.nn.Linear(64, 64)
        self.scale_factor = 1. / math.sqrt(64)
 
    def attention(self, x1, x2, x3):
        v1 = self.q(x1)
        v2 = self.k(x2)
        v3 = v1.matmul(v2.transpose(-2, -1))
        v4 = v3 * self.scale_factor
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.1)
        v7 = v6.matmul(x3)
        return v7
 
    def forward(self, x1, x2):
        v1 = self.attention(x1, x1, x1)
        v2 = self.attention(v1, x2, x2)
        return v2


# Initializing the model
m = Model(3)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
