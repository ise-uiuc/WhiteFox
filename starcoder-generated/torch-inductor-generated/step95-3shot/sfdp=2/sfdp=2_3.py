
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_qk = torch.nn.Dropout(0.3)
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2)
        v2 = v1.div(0.001)
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout_qk(v3)
        output = torch.mul(v4, x3)
        return v5, v6, v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 2, 3)
x3 = torch.randn(1, 2, 4)
x4 = torch.randn(1, 2, 4)
