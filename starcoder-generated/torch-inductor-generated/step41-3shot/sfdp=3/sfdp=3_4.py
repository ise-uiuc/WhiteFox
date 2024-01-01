
class Model(torch.nn.Module):
    def __init__(self, size_a, size_b, size_c, size_d):
        self.scale_factor = math.sqrt(size_c)
        self.mat_mul_dropout = nn.Dropout(0.5)
        super().__init__()
        self.linear_a = torch.nn.Linear(size_a, size_b)
        self.linear_b = torch.nn.Linear(size_b, size_c)
        self.linear_d = torch.nn.Linear(size_d, size_c)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = self.scale_factor * v1
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.mat_mul_dropout(v3)
        v5 = torch.matmul(x1, x2.transpose(-2, -1))
        v6 = v5.transpose(-2, -1)
        v7 = torch.matmul(v4, v6)
        v8 = self.linear_a(v8)
        v9 = self.linear_b(v7)
        v10 = v9 + v8
        v11 = self.linear_d(v7)
        return v10

# Initializing the model
m = Model(64, 64, 512, 256)

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.randn(1, 512, 256, 256)
