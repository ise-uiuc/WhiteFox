
class MyAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_projection=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_projection = output_projection
     
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax()
 
    def scale_factor_from_embed_dim(self):
        return math.sqrt(self.embed_dim)
 
    def forward(self, query, key, value, dropout_p=0.):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
     
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / self.scale_factor_from_embed_dim()
        softmax_qk = self.softmax(qk)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        if self.output_projection:
            output = self.output_projection(output)
        return output

# Initializing the model
m = MyAttention(8, 4)

# Inputs to the model
query = torch.randn(256, 8)
key = torch.randn(512, 8)
value = torch.randn(512, 8)
