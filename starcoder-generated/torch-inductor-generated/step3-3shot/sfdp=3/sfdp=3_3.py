
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 3.0
        v3 = self.softmax(v2)
        v4 = self.matmul(v3)
        v5 = torch.matmul(v4, x1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
