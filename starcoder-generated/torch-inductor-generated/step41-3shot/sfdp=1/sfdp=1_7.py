
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        s = 7
        self.scale_factor = 1 / math.sqrt(dim)
        self.matmul1 = torch.nn.Linear(dim, num_heads * s)
        self.matmul2 = torch.nn.Linear(num_heads * s, dim)
 
    def forward(self, x1, x2, _ = None):
        v1 = self.matmul1(x1)
        v2 = self.matmul2(x2)
        v3 = torch.matmul(x1, x2.transpose(-2, -1))
        v4 = v3 * self.scale_factor
        v5 = softmax(v4, dim=-1)
        v6 = drop_out(v5, 0.1)
        v7 = torch.matmul(v6.transpose(-2, -1), v2)
        return v7

# Initializing the model
m = Model(dim=128, num_heads=4)

# Inputs to the model
x1 = torch.randn(1, 128, 3)
x2 = torch.randn(1, 128, 6)
