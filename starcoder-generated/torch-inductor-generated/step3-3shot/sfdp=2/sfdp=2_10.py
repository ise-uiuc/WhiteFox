
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dot_product_attention = DotProductAttention(dropout=0.3, is_cross_attention=False)
        self.out = torch.nn.Linear(hidden_size, hidden_size)
 
    def forward(self, query, key, value, training=False):
        x = self.dot_product_attention(query=query, key=key, value=value, training=training)
        output = self.out(x)
        return output

# Initializing the model
m = Model(hidden_size=128)

# Inputs to the model
query = torch.randn(1, 128, 576)
key = torch.randn(1, 128, 576)
value = torch.randn(1, 128, 576)
