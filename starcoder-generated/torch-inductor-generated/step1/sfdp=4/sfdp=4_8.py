
def compute_attn_masks(input, attention_mask: torch.Tensor):
    attn_mask = None
    if len(attention_mask.size()) == len(input.size()):
        # attention_mask: [bsz, seq_len] -> [seq_len, bsz, seq_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Here we can assume input size equals to attention_mask.size() 
        # [seq_len, bsz, seq_len]
        attn_mask = (1.0 - attention_mask.flip([-1]).cumsum(-1).flip([-1])).masked_fill(attention_mask == 0, float('-inf'))
    elif len(attention_mask.size()) == len(input.size()) + 1:
        # attention_mask: [bsz x seq_len] -> [seq_len x bsz x seq_len]
        attention_mask = attention_mask.transpose(0, 1).unsqueeze(0)

        # Here we can assume input size equals to attention_mask.size()
        # [seq_len x bsz, seq_len]
        attn_mask = (1.0 - attention_mask.flip([-1]).cumsum(-1).flip([-1])).masked_fill(attention_mask == 0, float('-inf'))
    
    return attn_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, q, k, v, mask=True):
        super(ScaledDotProductAttention, self).__init__()
        self.masked = mask
        self.k = k  # [seq_len x bsz, n_heads, k]
        self.query = q  # [seq_len x bsz, n_heads, k]
        self.v = v  # [seq_len x bsz, n_heads, n_per_head]
    
    def forward(self, x):
        # [seq_len, bsz, n_heads, k] @ [n_heads, k, n_per_head] = [seq_len, bsz, n_heads, n_per_head]
        if self.masked:
            attn = (self.query @ self.k.transpose(-2, -1)) / math.sqrt(self.query.size(-1))
        else:
            attn = (self.query @ self.k.transpose(-2, -1))
        attn_mask = compute_attn_masks(x, None)
        if attn_mask is not None:
            attn += attn_mask
        attn = torch.softmax(attn, dim=-1)
        res = attn @ self.v
        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dim_per_head, mask=True):
        super(MultiHeadAttention, self).__init__()
        self.masked = mask
        self.dim_per_head = dim_per_head
        self.n_heads = n_heads
        self.d_model = d_model

        assert(d_model == n_heads * dim_per_head) 
        # 1) create a linear layer to split d_model into n_heads * dim_per_head. This will result in a tensor with shape [d_model, n_heads * dim_per_head].
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        # 2) create k, q, and v by linear transformations to d_model. These will result in tensors with shape [d_model, n_heads * dim_per_head].
        self.k = nn.Linear(d_model, n_heads * dim_per_head)
        self.q = nn.Linear(d_model, n_heads * dim_per_head)
        self.v = nn.Linear(d_model, n_heads * dim_per_head)
    
    def forward(self, x): 
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        d_model = self.d_model
        
        h = self.linear_layers[0](x)
        
        # Split [bsz, seq_len, n_heads * dim_per_head] into [bsz, seq_len, n_heads, dim_per_head]. So after linear transformation it will result in 3-dimensional tensor
        self.k, self.q, self.v = h.split([dim_per_head] * n_heads, dim=2)
        
        self.k = self.k.view(x.size(0), x.size(1), n_heads, dim_per_head)
        self.q = self.q.view(x.size(0), x.size(1), n_heads, dim_per_head)
        self.v = self.v.view(x.size(0), x.size(1), n_heads, dim_per_head)

        res = ScaledDotProductAttention(self.q, self.k, self.v)(x)
        # [bsz, seq_len, n_heads, dim_per_head] -> [bsz, seq_len, n_heads * dim_per_head]   
        res = res.contiguous().view(x.size(0), x.size(1), n_heads * dim_per_head)   
        res = self.linear_layers[1](res)
        return res

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_per_head, mask=True):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dim_per_head, mask)
        self.layer_norm_1 = LayerNorm(d_model)
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.layer_norm_2 = LayerNorm(d_model)
    
    def forward(self, x):
        res = self.multi_head_attention(x)
        x = self.layer_norm_1(x + res)
        feed_forward = F.relu(self.linear_layers[0](x))
        res = self.linear_layers[1](feed_forward)
        return self.layer_norm_2(x + res)


def PositionwiseFeedForward(x, d_inner_hid, d_hid):
    # After multi-head attention and before the addition there is a residual layer, 
    # which will simply add the output of the sub-layer to the output of the sub-layer,
    # with the sole exception of the first sub-layer in the layer, which does not have a residual connection
    return F.relu(x.transpose(1, 2) @ torch.nn.Linear(d_hid, d_inner_hid) @ torch.nn.Linear(d_inner_hid, d_hid).transpose(1, 2))



class TransformerModel(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dim_per_head, d_inner_hid, mask=True):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, dim_per_head, mask=mask) for _ in range(n_layers)])
        self.positionwise = PositionwiseFeedForward(0, d_inner_hid, d_hid)
    
    # (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.positionwise(x)
        return x.transpose(0, 1) 

# Initializing the model
n_layers = 2
d_model = 32
n_heads = 4
dim_per_head = 1
d_inner_hid = 64
mask = True
model = TransformerModel(n_layers, d_model, n_heads, dim_per_head, d_inner_hid, mask)

# Inputs to the model. 
# Input x should have dimensions [seq_len x bsz x d_model], 
# but for this model we will need x to have dimensions [bsz x seq_len x d_model]
x = torch.randn(128, 8, d_model)
