
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=1 - self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(3, 4, 8, 0.5, 0.1)

# Inputs to the model
query = torch.randn(2, 3, 4)
key = torch.randn(2, 4, 8)
value = torch.randn(2, 4, 8)
