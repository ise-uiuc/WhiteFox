
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.12500000000000001
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        return v4.matmul(x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 50)
x2 = torch.randn(2, 50, 768)
x3 = torch.randn(2, 768, 500)
