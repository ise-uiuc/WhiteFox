
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, x1, x2):
        m1 = torch.matmul(x1, x2.transpose(-2, -1))
        m2 = m1.div(1)
        m3 = m2.softmax(dim=-1)
        m4 = self.dropout(m3)
        v1 = m4.matmul(x2)
        return v1 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 32)
x2 = torch.randn(1, 16, 64)
