
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p1)
 
    def forward(self, x1):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = qk.div(inv_scale_factor)
        v2 = self.dropout(v1)
        v3 = v2.softmax(dim=-1)
        output = v3.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 8, 1)
x2 = torch.randn(4, 1, 8)
inv_scale_factor = torch.randn(1) / 128
x3 = torch.randn(4, 1, 8)
