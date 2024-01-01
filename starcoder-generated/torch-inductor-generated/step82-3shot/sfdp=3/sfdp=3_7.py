
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias1 = torch.nn.Parameter(torch.rand([1, 512]))
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.mul(0.2)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, 0.5)
        v5 = torch.matmul(v4, self.bias1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 768)
x2 = torch.randn(1, 512, 768)
