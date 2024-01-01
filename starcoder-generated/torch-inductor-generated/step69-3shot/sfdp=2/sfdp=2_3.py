
class Model(torch.nn.Module):
    def __init__(
        self,
        n_head = 1,
        dim_head = 128,
        dropout_p = 0.0
    ):
        super().__init__()
        self.dim_head = dim_head
        self.n_head = n_head
        self.dropout = torch.nn.Dropout(dropout_p)
 
        inner_dims = dim_head * n_head
        self.qkv = torch.nn.Linear(3072, 3 * inner_dims, bias=True)
 
    def forward(self, inp):
        qkv = self.qkv(inp).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1.0 / math.sqrt(v.size(-1))
        qk = qk.mul(scale_factor).softmax(dim=-1)
        qk = self.dropout(qk)
        output = torch.matmul(qk, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return output

# Initializing the model
m = Model()

# Inputs to the model
inp = torch.randn(1, 3072, 436)
