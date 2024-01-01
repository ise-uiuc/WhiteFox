
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(80*8, 80*8)
 
    def forward(self, x1, x2, x3, x4):
        qkkv = self.head(x1)
        qkkv = qkkv.reshape(1, 24, 80, 8)
        q = qkkv[:, :1,...]
        k = qkkv[:, 1:2,...]
        v = qkkv[:, 2:,...]
        qk = q @ k.transpose(-2, -1).matmul(1./math.sqrt(q.size(-1))).softmax(dim=-1)
        qkv = qk @ v
        return qkv

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 24*80*8) # The result of the query/key/value computation
x2 = torch.randn(1, 24*80*8)
x3 = torch.randn(1, 24*80*8)
x4 = torch.randn(1, 24*80*8) # The attention mask
