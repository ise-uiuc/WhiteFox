
class Model(torch.nn.Module):
    def __init__(self, query, key, scale_factor=1.0, dropout_p=0.0):
        super(Model, self).__init__()
        self.scaled_dot_product_attention = ScaledDotProductAttention(scale_factor=scale_factor, dropout_p=dropout_p)
 
    def forward(self, query, key, value):
        return self.scaled_dot_product_attention(query=query, key=key, value=value)

# Initializing the model
query = torch.rand(16, 32, 64)
key   = torch.rand(16, 32, 64)
value = torch.rand(16, 32, 64)
m = Model(query, key)

# Inputs to the model
