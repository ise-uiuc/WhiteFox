
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.3
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, query, key, value, query_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if query_mask is not None:
        qk += query_mask
        attn = qk / (query.size(-1) * value.size(-2))
        attn = self.dropout(attn)
        return attn @ value
        
# Initializing model
m = Model()

# Input tensors
query = torch.randn(1, query_seq_len, key_seq_len)
key = torch.randn(1, key_seq_len, query_seq_len)
value = torch.randn(1, key_seq_len, value_seq_len)
query_mask = torch.tril(torch.ones(query_seq_len, query_seq_len), diagonal=-1)
        
# Calculate output        
output = m(query, key, value, query_mask)
