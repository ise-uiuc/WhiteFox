
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        m1 = torch.matmul(x1, x2.transpose(-2, -1))
        m2 = m1.div(2.5)
        m3 = torch.matmul(m2, x4.transpose(-2, -1))
        m4 = m3.softmax(dim=-1)
        m5 = torch.nn.functional.dropout(m4, p=0.2)
        m6 = m5.matmul(x3)
        return m6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(6, 15, 7)
x2 = torch.randn(6, 15, 5)
x3 = torch.randn(6, 5, 7)
x4 = torch.randn(6, 15, 5)
