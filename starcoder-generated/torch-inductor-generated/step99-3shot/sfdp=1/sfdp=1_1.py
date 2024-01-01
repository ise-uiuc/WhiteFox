
class Model(torch.nn.Module):
    def __init__(self, batch, heads, length, depth):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.length = length
        self.depth = depth
        self.qk = torch.nn.Linear(depth, depth)
        self.v = torch.nn.Linear(depth, depth)
 
    def forward(self, x1, x2):
        qk = self.qk(x1)
        v = self.v(x2)
        q = qk.reshape([self.batch * self.heads, self.length, 1, self.length])
        k = qk.reshape([self.batch * self.heads, 1, self.length, self.length])
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        qk = torch.matmul(q, k)
        inv_scale_factor = math.sqrt(float(self.depth // self.heads))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        dropout_qk = dropout_qk.transpose(-2, -1)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(1, 8, 100, 256)

x1 = torch.randn(100, 256)
x2 = torch.randn(2, 1, 100, 256)
