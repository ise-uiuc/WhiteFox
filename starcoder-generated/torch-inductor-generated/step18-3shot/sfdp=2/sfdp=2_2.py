
class MultiHeadAttentionWithDropout(torch.nn.Module):
    def __init__(self, num_head, head_size, dropout_p=0.5):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(head_size, num_head)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v):
        v, att_scores = self.attention(q, k, v)
        v = self.dropout(v)
        return v, att_scores

# Initializing the model
num_head = 3
dropout_p = 0.3
m = MultiHeadAttentionWithDropout(num_head, num_head, dropout_p)

# Inputs to the model
query = torch.randn(128, 10, 16)
key = torch.randn(128, 11, 16)
value = torch.randn(128, 11, 16)
__output__, __scores__ = m(query, key, value)

