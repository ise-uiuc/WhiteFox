
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        output = v4.matmul(value)
        return output

# Initializing the model
m = Model(0.5000000000000001)

# Inputs to the model
query = torch.randn((1, 5, 1, 64))
key = torch.randn((1, 6, 1, 200))
value = torch.randn((1, 5, 1, 200))
