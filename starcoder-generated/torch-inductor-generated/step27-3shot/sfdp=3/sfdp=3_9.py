
class Model(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dk = dim
        self.scale_factor = 1.0 / math.sqrt(dim)

    def forward(self, inputs):
        query, key, value, dropout_p = inputs
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 64)
key = torch.randn(1, 12, 64)
value = torch.randn(1, 12, 64)
dropout_p = 0.5
