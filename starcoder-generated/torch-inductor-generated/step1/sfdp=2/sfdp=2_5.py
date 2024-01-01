
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        a1 = torch.matmul(query, key.transpose(-2, -1))
        a2 = a1 / inv_scale_factor
        a3 = torch.nn.functional.softmax(a2, dim=-1)
        v1 = torch.nn.functional.dropout(a3, p=self.dropout_p)
        v2 = torch.matmul(v1, value)
        return v2

# Initializing the model
m = Model(inv_scale_factor=10.0, dropout_p=0.5)

# Inputs to the model
query = torch.randn(8, 2, 4)
key = torch.randn(16, 3, 4)
value = torch.randn(16, 2, 3)
