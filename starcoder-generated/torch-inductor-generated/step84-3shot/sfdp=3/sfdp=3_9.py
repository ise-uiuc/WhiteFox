
class Model(torch.nn.Module):
    def __init__(self, dim, heads, dropout_p=0.5):
        super().__init__()
        self.dim_head = dim // heads
        self.heads = heads
        self.scale_factor = self.dim_head ** 0.5
        self.to_qk = torch.nn.Linear(dim, dim*2, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)
    
    def forward(self, input, mask=None):
        query, key, value = [l(x).view(x.size(0), x.size(1), self.heads, self.dim_head).transpose(2, 1) for l, x in zip(self.to_qk, (input, input, input))]
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value).transpose(2, 1).contiguous()
        return self.out_proj(output.view(output.size(0), output.size(1), self.dim))

# Initializing the model
m = Model(dim, heads)

# Input to the model
x1 = torch.randn(b, s, dim)
