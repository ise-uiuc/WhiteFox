
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, x1, x2):
        v = torch.matmul(x1, x2.transpose(-2, -1))
        v = v * 0.1
        v = self.softmax(v)
        v = self.dropout(v)
        v = torch.matmul(v, x2)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 64, 32)
x2 = torch.randn(2, 32, 64)
