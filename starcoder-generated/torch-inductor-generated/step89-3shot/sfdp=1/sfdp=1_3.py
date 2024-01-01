
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.inv_scale_factor = math.sqrt(self.head_dim ** -0.5)
    
    def forward(self, query, key, value):
        w = torch.matmul(query, key.transpose(-2, -1))
        v1 = w.div(self.inv_scale_factor)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=self.dropout_p)
        output = v3.matmul(value)
        return output

# Initializing the model
m = Model(head_dim=64)

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
