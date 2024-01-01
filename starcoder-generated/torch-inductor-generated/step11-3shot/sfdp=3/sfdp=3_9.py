
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.9):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        scale_factor = np.sqrt(query.size(-1))
        qk = query.matmul(key.transpose(-1, -2))
        v1 = qk.mul(scale_factor)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=self.dropout_p)
        return v3.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 15, 15)
key = torch.randn(1, 5, 15, 15)
value = torch.randn(1, 5, 10, 10)
