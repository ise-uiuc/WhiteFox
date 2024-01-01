
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 512
        self.dim = 64 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = torch.reshape(qk, (1, 8 * self.seq_len, 512 * self.seq_len)) # Reshape the model
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=2) # Apply softmax
        attn_weight = torch.reshape(attn_weight, (1, 8, self.seq_len, self.seq_len))
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = torch.cat([attn_weight for i in range(self.heads)], dim=1) # Output 7 attention layers
        output = output.reshape((1, 8 * self.seq_len, self.seq_len)) # Reshape the output layers
        ww = output @ value.transpose(-2, -1) # Compute the output layer
        output = ww.reshape((1, 7660, 512)) # Reshape the output layer
        return output
# Inputs to the model
query = torch.randn(1, 8, 512, 64)
key = torch.randn(1, 8, 512, 64)
value = torch.randn(1, 8, 512, 64)
attn_mask = torch.randn(1, 1, 512, 512)
