
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * scale
        v3 = F.softmax(v2, dim=-1)
        v4 = F.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Generating random inputs for the model
key = torch.randn(1, 125, 15)
query = torch.randn(1, 100, 125)
value = torch.randn(1, 1, 100)
scale = np.power(self.d_input, -0.5)
dropout_p = 0
