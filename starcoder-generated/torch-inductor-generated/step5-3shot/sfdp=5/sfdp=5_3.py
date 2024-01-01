
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=self.dropout_p)
    def forward(self, x1):
        v1, v2 = self.attention(queries=x1, key=x1, value=x1, key_padding_mask=x1)
        return v1

# Initializing the model
x1 = torch.randn(1, 64, 704)
m = Model()
with torch.no_grad():
    