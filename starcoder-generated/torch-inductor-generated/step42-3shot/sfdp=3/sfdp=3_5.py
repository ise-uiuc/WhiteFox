
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        # The dimension must be divisible by the number of heads.
        self.dim_per_head = dim // num_heads
        self.sqrt_dim_per_head = math.sqrt(self.dim_per_head)
        self.query = Parameter(torch.Tensor(dim, dim))
        self.key = Parameter(torch.Tensor(dim, dim))
        self.value = Parameter(torch.Tensor(dim, dim))
        self.softmax_scale = math.sqrt(math.sqrt(dim))
        self.dropout = nn.Dropout(dropout_p)
 
    def forward(self, x1, x2):
        q = x1.matmul(self.query.div_(self.sqrt_dim_per_head))
        k = x2.matmul(self.key.div_(self.sqrt_dim_per_head)).transpose(-2, -1)
        k = self.softmax_scale * k
        v = x2.matmul(self.value.div_(self.sqrt_dim_per_head))
        qktv = q.matmul(k).matmul(v)
        result = self.dropout(qktv)
        return result

# Initializing the model
m = Model(3072, 2, 0.1)

# Inputs to the model
x1 = torch.randn(20, 25, 3072)
x2 = torch.randn(20, 25, 3072)
