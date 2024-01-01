
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.2
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        v5 = v4.matmul(x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 16, 64)
x2 = torch.randn(1, 3, 64, 16)
x3 = torch.randn(1, 3, 16, 16)
