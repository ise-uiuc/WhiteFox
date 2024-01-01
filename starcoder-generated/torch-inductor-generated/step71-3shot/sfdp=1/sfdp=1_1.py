
def MultiHeadSelfAttention(in_dim, num_attn_heads, out_dim, dropout_p):
  
    def __init__(self):
        super().__init__()
        self.W_Q = Linear(in_dim, num_attn_heads * out_dim)
        self.W_K = Linear(in_dim, num_attn_heads * out_dim)
        self.W_V = Linear(in_dim, num_attn_heads * out_dim)
 
    def forward(self, query, key, value, mask=None):
        qHead = self.W_Q(query)
        kHead = self.W_K(key)
        vHead = self.W_V(value)
 
        qHead = qHead.view(size=qHead.size(0),
                            num_heads=num_attn_heads,
                            size=query.size(1),
                            size=out_dim)
        kHead = kHead.view(size=kHead.size(0),
                            num_heads=num_attn_heads,
                            size=key.size(1),
                            size=out_dim)
        vHead = vHead.view(size=vHead.size(0),
                            num_heads=num_attn_heads,
                            size=value.size(1),
                            size=out_dim)
 
        attnOut = torch.matmul(qHead.transpose(2, 3), kHead)
        scaleFactor = torch.sqrt(torch.tensor(size=qHead.size()[-2:]))
        scaledAttnOut = attnOut / scaleFactor[None, None, :, :]
 
        if mask is not None:
            scaledAttnOut = scaledAttnOut.masked_fill_(mask, -1e9)
 
        softMaxAttnOut = scaledAttnOut.softmax(dim=-1)
 
        dropoutAttnOut = torch.nn.dropout(softMaxAttnOut, p=dropout_p)
 
        out = torch.matmul(dropoutAttnOut, vHead)
 
        out = out.view(size=out.size(0),
                        size=out.size(1),
                        num_heads=num_attn_heads * out_dim)
 
        return out

# Initializing the model
m = MultiHeadSelfAttention(in_dim=512,
                            num_attn_heads=8,
                            out_dim=64,
                            dropout_p=0.2)

# Inputs to the model
query = torch.randn(1, 1024, 512)
key = torch.randn(1, 1024, 512)
value = torch.randn(1, 1024, 512)
mask = torch.zeros(query.size(1), key.size(1))
