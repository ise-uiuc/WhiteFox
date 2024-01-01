
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.8)
 
    def forward(self, x1, x2, x3, x4):
        q = self.mat(x1)
        k = self.mat(x2)
        k = k.T
        v = self.mat(x3)
        qk = torch.matmul(q, k)
        inv_scale_factor = torch.sqrt(torch.as_tensor(q.size(-1)).float()).reciprocal()
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        drop_qk = self.dropout(softmax_qk)
        o = torch.matmul(drop_qk, v)
        return o, o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)


