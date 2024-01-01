
class Attention(torch.nn.Module):
    def __init__(self, query_size, key_size, attn_mask, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.attn_mask = attn_mask
 
    def forward(self, query, key, value):
        # query: [batchsize, query_size, query_len]
        # key: [batchsize, key_size, key_len]
        # value: [batchsize, value_size, value_len]
        # attn_mask: [batchsize, 1, query_len, key_len]
        qk_out = torch.matmul(query, key.transpose(-2, -1))
        qk_out = qk_out / math.sqrt(query.size(-1))
        qk_out = (qk_out + self.attn_mask) * math.sqrt(query.size(-1))
        attn_weight = torch.softmax(qk_out, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, True)
        attn_output = torch.matmul(attn_weight, value)
        return attn_output

# Initializing the model
attn = Attention(10, 20, 0.5, torch.randn(5, 1, 9, 12))

# Inputs to the model
query = torch.randn(5, 10, 9)
key = torch.randn(5, 20, 12)
value = torch.randn(5, 30, 12)
attn_mask = tensor.ones(5, 1, 9, 12)
