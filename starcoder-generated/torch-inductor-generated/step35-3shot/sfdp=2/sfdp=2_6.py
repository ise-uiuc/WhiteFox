
class Model(torch.nn.Module):
    def __init__(self, d_query, d_key, d_value, dropout_p=0.2):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(d_query))
        self.key = torch.nn.Parameter(torch.empty(d_key))
        self.value = torch.nn.Parameter(torch.empty(d_value))
        self.scale_factor = 1.0 / np.sqrt(d_query)  # Scale the dot product by this factor
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        out = torch.matmul(query, key.transpose(-2, -1))
        out = out * self.scale_factor
        out = out.softmax(dim=-1)
        out = self.dropout(out)
        out = torch.matmul(out, value)
        return out
 
# Initializing the model
m = Model(256, 256, 256)

# Inputs to the model
query = torch.randn(16, 256)
key = torch.randn(16, 256)
value = torch.randn(16, 256)
