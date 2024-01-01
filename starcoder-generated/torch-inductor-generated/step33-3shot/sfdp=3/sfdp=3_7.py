
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
 
    def forward(self, x1, x2, x3):
        b1, l1, e1 = x1.size()
        b2, l2, e2 = x2.size()
        b3, l3, e3 = x3.size()
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * e2 ** -0.5
        v3 = F.softmax(v2, dim=-1)
        v4 = F.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(48, 10, 64)
x2 = torch.randn(48, 10, 64)
x3 = torch.randn(48, 10, 64)
y = m(x1, x2, x3)
print(y.size())

