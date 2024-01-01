
class SelfAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, heads, dropout):
        super().__init__()
        self.dim_out = dim_out
        self.heads = heads
        self.scale_factor = dim_out // heads
        self.to_q = torch.nn.Linear(dim_in, dim_out, bias=False)
        self.to_k = torch.nn.Linear(dim_in, dim_out, bias=False)
        self.to_v = torch.nn.Linear(dim_in, dim_out, bias=False)
        self.to_out = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x):
        num_heads, b, _ = self.heads, x.shape[0], x.shape[-1]

        q = self.to_q(x).softmax(-1).unsqueeze(-3)
        k = self.to_k(x).softmax(-2).transpose(-2, -1).unsqueeze(-3)
        v = self.to_v(x).transpose(-2, -1).unsqueeze(-3)

        q = torch.cat(torch.unbind(q, dim=-3), dim=-2)
        k = torch.cat(torch.unbind(k, dim=-3), dim=-2)
        v = torch.cat(torch.unbind(v, dim=-3), dim=-2)

        output = self.dropout(torch.matmul(q, v.transpose(-2, -1)) / math.sqrt(self.scale_factor))
        output = torch.matmul(output, k)
        output = output.apply(lambda x: x / x.shape[-1])
        y = torch.cat(torch.unbind(output, dim=-3), dim=-1)
        return self.to_out(y)

num_heads = 4
model = SelfAttention(16, 64, num_heads, 0.2)
x = torch.randn(2, 128, 16)
model(x)

# Initializing the model
m = AttentionModel()

# Inputs to the model
x = torch.randn(1, 64, 64, 64)
