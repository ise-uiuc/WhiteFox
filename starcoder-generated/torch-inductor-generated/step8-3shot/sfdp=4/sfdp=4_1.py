
class Model(torch.nn.Module):
    def __init__(self, n_head, d_kv):
        super().__init__()
        self.n_head = n_head
        self.d_kv = d_kv

    def forward(self, query, key, value, attn_mask):
        # Create the heads projection layers if necessary
        if not hasattr(self, 'q_proj'):
            self.q_proj = torch.nn.Linear(query.size(-1), self.n_head * self.d_kv)
            self.k_proj = torch.nn.Linear(key.size(-1), self.n_head * self.d_kv)
            self.v_proj = torch.nn.Linear(value.size(-1), self.n_head * self.d_kv)
        if not hasattr(self, 'o_proj'):
            self.o_proj = torch.nn.Linear(self.n_head * self.d_kv, value.size(-1))
        # Reshape the query, key and value tensors into N heads
        q = self.q_proj(query).view(query.size(0), query.size(1), self.n_head, self.d_kv).transpose(1, 2)
        k = self.k_proj(key).view(key.size(0), key.size(1), self.n_head, self.d_kv).transpose(1, 2)
        v = self.k_proj(value).view(key.size(0), key.size(1), self.n_head, self.d_kv).transpose(1, 2)
        # Compute the scaled dot product between the query and key tensors
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask # Add the attention mask
        # Apply softmax to the scaled dot product to get the attention probabilities
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v # Compute the weighted tensor
        # Reshape the weighted tensor back to its original dimensions
        output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(1), value.size(-1))
        # Apply the projection layer to the reshaped weighted tensor to produce the final output
        output = self.o_proj(output)
        return output

# Initializing the model
n_head = 4
d_kv = 128
m = Model(n_head, d_kv)

# Inputs to the model
query = torch.randn(1, 16, 128)
key = torch.randn(1, 32, 128)
value = torch.randn(1, 32, 128)
x1 = torch.randn(1, 16, 128) # The attention mask for masked language model is a tensor with -1000 where the attention should be skipped
