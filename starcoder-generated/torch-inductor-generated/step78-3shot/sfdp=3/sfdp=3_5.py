
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.functional.dropout
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.5
        v3 = self.p(v2, p=0.5)
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
