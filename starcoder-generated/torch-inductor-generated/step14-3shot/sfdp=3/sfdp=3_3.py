
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, x1, x2, x3):
        m1 = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = m1 * 1.0
        a1 = torch.nn.functional.softmax(v1, dim=-1)
        d1 = self.dropout(a1)
        v2 = x3 * 1.0
        v3 = d1.matmul(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12, 30, 20)
x2 = torch.randn(1, 12, 30, 20)
x3 = torch.randn(1, 12, 12, 10)
