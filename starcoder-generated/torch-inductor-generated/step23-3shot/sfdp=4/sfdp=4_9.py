
class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads" 
        self.scaling = self.head_dim ** -0.5
        self._qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self._out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
  
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False): 
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim, f"query.size(-1) must be equal to embed_dim. {query.size(-1)}!= {self.embed_dim}"
        assert key.size() == value.size(), "key must be equal to value. "
        src_len, _, _ = key.size()
        assert src_len, bsz == value.size(0), value.size(1) 

        qkv = self._qkv_proj(query) 
        qkv = qkv.reshape(tgt_len, bsz * self.num_heads, 3 * self.head_dim).transpose(0, 1) 
        query, key, value = qkv.split([self.head_dim, self.head_dim, self.head_dim], dim=-1) 
        attn_weight = (query @ key.transpose(-2, -1)) * self.scaling 
        attn_weight_float = torch.where(attn_weight == 0, torch.tensor(float("-Inf")).to(torch.double), attn_weight) 

        assert attn_mask is not None, "The attention mask must not be None" 
        attn_mask = torch.where(attn_mask == 0, 0, attn_mask * -10000.0) 
        attn_weight = attn_weight_float + attn_mask 

        assert key_padding_mask is not None, "The attention key padding mask must not be None" 
        attn_weight = torch.where(key_padding_mask.transpose(-2, -1) == 0, attn_weight, torch.tensor(float("-Inf")).to(torch.double)) 

        attn_weight = torch.softmax(attn_weight, dim=-1) 
        attn_weight = self.dropout(attn_weight) 
        attn_output = attn_weight @ value 
        attn_output = attn_output.transpose(0, 1).reshape(bsz, tgt_len, self.embed_dim) 
        attn_output = self._out_proj(attn_output)
        return attn_output 
    
# The forward method of TransformerEncoder should take a batch of masks in which all elements are either `True` or `False` and all dimensions has the same shape as a tensor after the mask has been applied to the tensor.

m = MultiheadAttention(embed_dim=16, num_heads=2) 
tgt = torch.randn(20, 32, 16)
src = torch.randn(20, 32, 16)
mask = torch.randn(20, 32) > 0 
mask = mask[:, None, None, :]
y = m(tgt, src, src, attn_mask=mask, need_weights=False)
assert y.shape == (32, 20, 16) 

