

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.nn.Linear(64, 128)
        self.mat2 = torch.nn.Linear(64, 128)
        self.mat3 = torch.nn.Linear(128, 128)
 
    def forward(self, x1, x2):
        v1 = self.mat1(x1)
        v2 = self.mat2(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3 / math.sqrt(v1.shape[-1])
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=dropout_p)
        v7 = torch.matmul(v6, x1)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
