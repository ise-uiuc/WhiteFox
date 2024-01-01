
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.head = SingleHeadAttention(embed_dim, num_heads, dropout_p)
        self.layernorm = torch.nn.LayerNorm([embed_dim])
 
    def forward(self, query, key, value):
        output = self.head(query, key, value)
        output = self.layernorm(output)
        return output

# Initializing the model
m = Model(embed_dim=128, num_heads=4, dropout_p=0.5)

# Inputs to the model
query = torch.randn(2, 3, 128)
key = torch.randn(2, 4, 128)
value = torch.randn(2, 4, 128)
