
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.matmul1 = torch.nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.matmul2 = torch.nn.Linear(hidden_size, 1)
 
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # 4-D tensor
        x = x.view(*new_x_shape) 
        return x.permute(0, 2, 1, 3) # 4-D tensor

    def forward(self, query, key, value, mask):
        # Self-attention pattern
        self.num_attention_heads = 2
        self.attention_head_size = 16
        q = self.matmul1(query)
        k = self.matmul1(key)
        v = self.matmul1(value)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        # Compute the dot product of the query and key, and scale it
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # Add the mask to the scaled dot product
        qk = qk + mask
        # Apply softmax to the result
        attn_weight = torch.softmax(qk, dim=-1)
        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v
        return output, attn_weight

# Initializing the model
m = Model(23)

# Inputs to the model
query = torch.randn(2, 6, 3, 23)
key = torch.randn(2, 7, 3, 23)
value = torch.randn(2, 7, 3, 23)
mask = torch.zeros_like(query)
mask[0, 0, :, 0] = float('-inf')
mask[0, 2, :, 1] = float('-inf')
