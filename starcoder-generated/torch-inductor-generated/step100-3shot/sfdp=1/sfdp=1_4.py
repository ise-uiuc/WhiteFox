
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=2, dropout_p=0.66):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim

        # Query, key, and value maps
        self.query = torch.nn.Linear(dim, dim, bias=True)
        self.key = torch.nn.Linear(dim, dim, bias=True)
        self.value = torch.nn.Linear(dim, dim, bias=True)

    def forward(self, query, key, value, *args, mask=None, **kwargs):
        n, c, h = self.num_heads, self.dim, query.shape[1]
    
        # Compute the query, key, and value tensors
        q = self.query(query).view(n, c, h).transpose(-2, -1).unsqueeze(1)
        k = self.key(key).view(n, c, h).transpose(-2, -1).unsqueeze(1)
        v = self.value(value).view(n, c, h).transpose(-2, -1).unsqueeze(1)
    
        # Compute the dot product and scale by the inverse square root of the attention dimension
        attn = torch.matmul(q, k.transpose(-2, -1)) / (c**0.5)

        # Add the mask
        if mask is not None and mask.numel() > 0:
            mask = mask.repeat(n, 1, 1)
            attn = attn.masked_fill(mask == 0, -1e+38)
    
        # Apply softmax
        attn = attn.softmax(dim=-1)
    
        # Apply dropout and compute the context tensor
        attn = torch.nn.functional.dropout(attn, p=0.66, training=self.training)
        ctxt = attn.matmul(v).squeeze(1)
    
        # Reshape the context tensor for downstream processing
        batch_size, num_query, dim = query.size(0), query.size(1), self.dim
        output = ctxt.contiguous().view(batch_size, num_query, h*c)
    
        return output, attn

# Initializing the model
m = Attention(dim=64)

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 32, 64)
x3 = torch.randn(1, 32, 64)
__output__, __att__ = m(x1, x2, x3)




