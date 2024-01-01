
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0.6
        self.eps = 1e-10
        self.q = torch.nn.Linear(64, 8)
        self.k = torch.nn.Linear(512, 8)
        self.v = torch.nn.Linear(512, 64)
        self.dropped_v = torch.nn.Dropout(self.p)
 
    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        inv_sf = q.size(-1) ** -0.5
        
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * inv_sf
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropped_v(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 512, 8)
