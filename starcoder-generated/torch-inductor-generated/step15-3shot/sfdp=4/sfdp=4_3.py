
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def masked_attention(self, q, k, v, attn_mask):
        # Compute the dot product of the query and key, and scale it
        q_k = torch.matmul(q, k)
        q_k = torch.div(q_k, math.sqrt(q.shape[-1]))
        # Add the attention mask to the scaled dot product
        q_k = q_k + attn_mask
        # Apply softmax to the result
        q_k = torch.softmax(q_k, dim=-1)
        return torch.matmul(q_k, v)

    def forward(self, q, k, v, attn_mask):
        v_ = self.masked_attention(q, k, v, attn_mask)
        return v_

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(4, 8, 32)
k = torch.randn(4, 24, 32)
v = torch.randn(4, 24, 32)
attn_mask = torch.randn(4, 32)
