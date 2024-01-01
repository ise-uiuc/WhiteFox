
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        v1 = inv_scale_factor * qk
        v2 = v1.softmax(dim=-1)
        v3 = self.dropout(v2)
        output = torch.matmul(v3, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
key = torch.randn(1, 6, 64, 128)
value = torch.randn(1, 6, 64, 128)
query = torch.randn(1, 6, 64, 128)
