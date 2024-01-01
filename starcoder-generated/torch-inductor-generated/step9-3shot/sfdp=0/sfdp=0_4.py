
class Model(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.query = torch.nn.Linear(dim1, dim2, bias=False)
        self.key = torch.nn.Linear(dim1, dim2, bias=False)
        self.value = torch.nn.Linear(dim1, dim2, bias=False)
        self.proj = torch.nn.Linear(dim2, dim1, bias=False)
 
    def forward(self, q, k, v):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / query.shape[-1]**-0.5
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        output = self.proj(output)
        return output

# Initializing the model
m = Model(dim=128, dim1=256)

# Inputs to the model
q = torch.randn(1, 10, 128)
k = torch.randn(1, 5, 128)
v = torch.randn(1, 5, 128)
