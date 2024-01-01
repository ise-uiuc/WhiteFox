
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k, dropout_p=0.5):
        super().__init__()
        self.inv_scale = 1/(d_k ** 0.5)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        attn = torch.matmul(query, key.transpose(-2, -1))
        attn = self.dropout(attn)
        attn = attn.softmax(-1)
        attn = attn.matmul(value)
        return attn

# Initializing the model
attn_m = ScaledDotProductAttention(16, 0.2)

# Inputs to the model
query = torch.randn(1, 10, 16)
key = torch.randn(1, 10, 16)
value = torch.randn(1, 10, 16)
