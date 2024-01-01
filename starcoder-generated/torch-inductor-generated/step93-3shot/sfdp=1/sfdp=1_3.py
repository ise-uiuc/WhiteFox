
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul0 = torch.nn.functional.matmul
 
    def forward(self, x1, x2, x3):
        v1 = self.matmul0(x1, x2.transpose(-2, -1))
        v2 = v1.div(5.0)
        v3 = F.softmax(v2, dim=-1)
        v4 = F.dropout(v3, p=None)
        v5 = v4.matmul(x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 5)
x3 = torch.randn(5, 6)
