
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.ones(4, 4))
        self.key = torch.nn.Parameter(torch.ones(4, 4))
        
    def forward(self, x1):
        v1 = torch.matmul(self.query, self.key.transpose(-2, -1))
        v2 = v1 * 2.0
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, x1)
        return v5

# Initializing the model
m = Model()

# Generate input tensor
x1 = torch.randn(4, 4)
