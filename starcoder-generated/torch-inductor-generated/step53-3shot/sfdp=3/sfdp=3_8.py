
class Model(torch.nn.Module):
    def __init__(self, query_embed_dim, key_embed_dim, num_heads, dropout_p=0, bias=True):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.FloatTensor([key_embed_dim/num_heads]))
        
        self.W_query = torch.nn.Linear(query_embed_dim, key_embed_dim, bias=bias)
        self.W_key = torch.nn.Linear(key_embed_dim, key_embed_dim, bias=bias)
        self.W_value = torch.nn.Linear(key_embed_dim, value_embed_dim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        q = self.W_query(query)
        k = self.W_key(key)
        v = self.W_value(value)
        
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = self.dropout(softmax_qk)
        
        return torch.matmul(dropout_qk, v)

# Initializing the model
m = Model(key_embed_dim=512, query_embed_dim=512, num_heads=2, dropout_p=0.2, bias=True)

# Inputs to the model
query1 = torch.randn(5, 10, 512)
key1 = torch.randn(5, 10, 512)
value1 = torch.randn(5, 10, 512)
