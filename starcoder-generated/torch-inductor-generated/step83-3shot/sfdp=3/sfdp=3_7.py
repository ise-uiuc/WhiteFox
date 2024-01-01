
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(dim_hidden, dim_hidden))
        self.dropout = 0.1
 
def scaled_dot_product(q, k):
    return torch.matmul(q, k.T)*dim_hidden**-0.5
 
def mlp(x):
    return torch.nn.functional.gelu(torch.matmul(x, self.w))
 
def forward(self, q, k, v, mask=None):
    scale_factor = (dim_hidden)^-0.5
    attn = scaled_dot_product(q, k)*scale_factor
 
    if mask is not None: # Put -inf to padding positions to disable them.
        mask = mask.to(torch.bool)
        attn = attn.masked_fill(mask == 0, -np.inf)
 
    attn_drop = torch.nn.functional.dropout(attn, p=self.dropout)
    out = mlp(attn_drop)
    return out
 
def get_attn_mask(seq_len, is_training=False):
    if is_training:
        attn_mask = []
        for line in range(seq_len):
            line = [0.0 for i in range(line)]
            line +=  [float('-inf') for i in range(seq_len - line)]
            attn_mask.append(line)
        return attn_mask
    else:
        return None
 
m = Model()

# Inputs to the model
q = torch.randn(seq_len, dim_hidden)
k = torch.randn(seq_len, dim_hidden)
v = torch.randn(seq_len, dim_hidden)
mask = get_attn_mask(seq_len)
