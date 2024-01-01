
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = query.size(-1) ** -0.5
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0)
        dropout_qk = dropout_qk.mul(dropout_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
