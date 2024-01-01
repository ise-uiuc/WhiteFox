
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, x1, x2, x3):
        s1 = torch.matmul(x1, x2.transpose(-2, -1)).div(inv_scale_factor)
        s2 = self.dropout(s1.softmax(dim=-1))
        y1 = s2.matmul(x3)
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
