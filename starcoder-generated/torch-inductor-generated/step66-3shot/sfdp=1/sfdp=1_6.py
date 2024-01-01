
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_qk = torch.nn.Dropout(0.5) # p = 0.5
 
    def forward(self, x1, x2):
        v = torch.matmul(x1, x2.transpose(-2, -1))
        v = v.div(10.0)
        v = self.dropout_qk(v)
        v = torch.matmul(v, x2)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(2, 10, 56, 56)
