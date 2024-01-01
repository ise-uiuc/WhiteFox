
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(n, c_q, h, w)
x2 = torch.randn(n, c_k, h, w)
x3 = torch.randn(n, c_v, h, w)
x4 = torch.Tensor.new(n, M).uniform_(1, 1)
