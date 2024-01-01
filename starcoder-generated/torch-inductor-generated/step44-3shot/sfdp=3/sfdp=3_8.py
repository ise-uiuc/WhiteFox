
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        weighted_qk = qk * scale_factor
        dropout_qk = torch.nn.functional.dropout(weighted_qk.softmax(dim=-1), p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 16, 32)
key = torch.randn(1, 1, 16, 32)
value = torch.randn(1, 1, 16, 32)
scale_factor = torch.Tensor([1.0/math.sqrt(32)])
