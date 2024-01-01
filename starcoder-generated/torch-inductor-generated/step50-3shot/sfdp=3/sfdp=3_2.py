
class Model(torch.nn.Module):
    def softmax_for_matmul(self, x1):
        v1 = torch.matmul(x1, x1.transpose(-2, -1))
        v2 = v1.mul(0.5)
        return v2.softmax(dim=-1)

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x1, x2):
        v1 = self.softmax_for_matmul(x1)
        v2 = self.dropout(v1)
        return v2.matmul(x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 5)
x2 = torch.randn(2, 5, 6)
