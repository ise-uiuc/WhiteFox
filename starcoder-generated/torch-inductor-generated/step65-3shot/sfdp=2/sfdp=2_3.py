
class Model(torch.nn.Module):
    def __init__(self, d_q, d_k, dropout_p=0.2):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scale_factor = (d_k) ** (-0.25)
 
    def forward(self, query, key, value):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        output = torch.matmul(v4, value)
        return output

# Initializing the model
d_q, d_k = 32, 32
model = Model(d_q, d_k)

# Inputs to the model
query = torch.randn(1, 1000, d_q)
key = torch.randn(1, 1000, d_k)
value = torch.randn(1, 1000, d_v)
__output1__ = model(query, key, value)

# Inputs to the model
query = torch.randn(1, 10, d_q)
key = torch.randn(1, 1000, d_k)
value = torch.randn(1, 1000, d_v)
__output2__ = model(query, key, value)

