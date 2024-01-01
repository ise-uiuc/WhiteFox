
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        v0 = query.matmul(key.transpose(-2, -1))
        v1 = v0 * scale_factor
        v2 = F.softmax(v1, dim=-1)
        v3 = F.dropout(v2, p=dropout_p)
        v4 = v3.matmul(value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 128)
key = torch.randn(1, 128, 64)
value = torch.randn(1, 128, 64)
scale_factor = 1.0 / math.sqrt(128)
dropout_p = 0.0
