
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dot_product = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = self.dot_product(v2)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 100, 10)
key = torch.randn(1, 200, 10)
