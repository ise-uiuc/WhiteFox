
class MultiheadAttention(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        scaled_dot_product = torch.matmul(query / math.sqrt(self.input_shape), key.transpose(-2, -1))
        scores = nn.functional.dropout(scaled_dot_product, p=self.dropout, training=training) # Apply dropout to the softmax output
        attn_weights = torch.softmax(scores, dim=-1, dtype=at.float) # Apply softmax to the result
        output = torch.matmul(attn_weights, value)
        self(query, key, value, attn_weights)
        return output

# Initializing the model
m = MultiheadAttention(d_model=512, num_heads=8, dropout=0.1)

query = torch.randn(8, 60, 512)
key = torch.randn(8, 100, 512)
value = torch.randn(8, 100, 512)

# Inputs to the model
