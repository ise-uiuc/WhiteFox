
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, scale_factor):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.mul(scale_factor)
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        output = torch.matmul(v4, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 4)
key = torch.randn(2, 3, 4)
value = torch.randn(2, 3, 4)
scale_factor = 2.5
dropout_p = 0.4
