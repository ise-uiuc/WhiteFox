
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.w_q = torch.nn.Linear(64, 64 * self.num_heads)
        self.w_k = torch.nn.Linear(64, 64 * self.num_heads)
        self.w_v = torch.nn.Linear(64, 64 * self.num_heads)
        self.dropout_1 = torch.nn.Dropout(p=0.2)
 
    def forward(self, q, k, v):
        self.q = self.w_q(q)
        self.k = self.w_k(k)
        self.v = self.w_v(v)
        self.transpose_q = self.q.transpose(dim0=2, dim1=3)
        self.transpose_k = self.k.transpose(dim0=2, dim1=3)
        self.transpose_v = self.v.transpose(dim0=2, dim1=3)
        self.softmax_qk = torch.softmax(torch.matmul(self.transpose_q, self.transpose_k.transpose(0, 1)) / math.sqrt(64), dim=-1)
        self.dropout_qk = self.dropout_1(self.softmax_qk)
        self.dropout_v = self.dropout_qk.matmul(self.transpose_v)
        return self.dropout_v.transpose(dim0=2, dim1=3)

model = Model(num_heads=3)
q = torch.randn(10, 30, 64)
k = torch.randn(8, 20, 64)
v = torch.randn(12, 20, 64)
print(model.forward(q, k, v).shape)

