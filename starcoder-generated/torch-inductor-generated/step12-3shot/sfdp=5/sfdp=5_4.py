
class Model(torch.nn.Module):
    # query, key, value are of shape: (N, heads, seq_len, dim) 
    # attn_mask is of shape: (heads, seq_len, seq_len)
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 512
        self.dim = 64 // self.heads

    def forward(self, query, key, value, attn_mask):
        # Apply the dot product between the query and key (plus an attention mask), and then divide by the square root of the dimension specified by the length of the query
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
 
        # Since the length of the attention mask is shorter than the actual size of the query and key, some rows will be added to the attention mask to adapt to the size of the inputs. These rows will correspond to the values `1e-10`
        qk = qk + attn_mask
 
        # Apply softmax to the scaled dot product of the query and key (plus the attention mask) to obtain the attention weights
        attn_weight = torch.softmax(qk, dim=-1)
 
        # Apply dropout to the attention weights
        attn_weight = torch.dropout(attn_weight, 0.1, True)
 
        # After that, compute the dot product of the attention weights and value to output the results
        output = attn_weight @ value
 
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 512, 64)
key = torch.randn(1, 8, 512, 64)
value = torch.randn(1, 8, 512, 64)
attn_mask = torch.randn(1, 1, 512, 512)
