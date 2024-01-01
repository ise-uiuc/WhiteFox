
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.inv_scale_factor = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
 
    def forward(self, q, k, v, mask=None, dropout_p=0.5):
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value
        return output

# Inputs to the model
query = torch.randn(2, 2, 10)
key = torch.randn(2, 4, 10)
value = torch.randn(2, 4, 10)
attention = ScaledDotProductAttention(embed_dim=10)
mask = torch.zeros(2,2).to(torch.bool)
dropout_p = 0.5
__outputs__ = attention(query, key, value, mask, dropout_p)

# Check if the outputs are correct
torch.equal(__outputs__, torch.ones(2, 10))

