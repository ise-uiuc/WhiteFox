
class Model(torch.nn.Module):
    def __init__(self, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = 64 / self.num_heads
        self.proj_q = torch.nn.Parameter(torch.Tensor(self.num_heads, self.head_size, self.head_size))
        self.proj_k = torch.nn.Parameter(torch.Tensor(self.num_heads, self.head_size, self.head_size))
        self.proj_v = torch.nn.Parameter(torch.Tensor(self.num_heads, self.head_size, self.head_size))
 
    def forward(self, q, k, v, mask):
        a = torch.matmul(q, self.proj_k.transpose(0, 1))
        b = a / (self.head_size ** 0.5)
        c = torch.softmax(b, dim=-1)
        d = torch.matmul(c, self.proj_v)
        dropout_d = torch.nn.functional.dropout(d, 0.1,  self.training)
        if mask is not None:
            dropout_d = dropout_d.masked_fill(mask.unsqueeze(1), float('-inf'))
        e = torch.matmul(dropout_d, self.proj_v.transpose(0, 1))
        return e

# Initializing the model
m = Model(num_heads=16)

# Inputs to the model
q = torch.randn(1, 16, 512, 64)
k = torch.randn(1, 16, 512, 64)
v = torch.randn(1, 16, 512, 64)
mask = torch.randn(1, 1, 512) > 0
