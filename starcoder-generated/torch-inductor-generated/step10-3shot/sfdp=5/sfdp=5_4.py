
class Model(torch.nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=8, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_p = dropout_p
 
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_weight = torch.matmul(q, k.transpose(-2, -1))
        attn_weight /= math.sqrt(q.size(-1))
        if mask is not None:
            attn_weight += mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, True)
        output = torch.matmul(attn_weight, v)
        return output
 
    def forward(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        mask = torch.tensor(tril_ones, dtype=torch.float32)
        mask = mask.view(1, 1, 8, 8)
        out = self.scaled_dot_product_attention(q, k, v, mask)
        return out

# Initializing the model
hidden_size = 64
num_attention_heads = 2
dropout_p = 0.1
m = Model(hidden_size, num_attention_heads, dropout_p)

# Inputs to the model
seq = torch.randn(1, 3, hidden_size)
