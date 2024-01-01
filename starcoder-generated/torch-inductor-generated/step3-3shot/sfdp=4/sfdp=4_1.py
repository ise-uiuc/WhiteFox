
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
 
    def forward(self, x1):
        output, attn_weight = self.attention(x1, x1, x1)
        return output

# Initializing the model
m = MultiHeadAttention(128, 16)

# Inputs to the model
x1 = torch.randn(8, 32, 128)
