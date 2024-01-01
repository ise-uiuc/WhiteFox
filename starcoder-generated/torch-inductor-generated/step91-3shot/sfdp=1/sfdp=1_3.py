
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p):
        inv_scale_factor = torch.pow((1.0 - dropout_p), 0.5)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        return scaled_qk.softmax(dim=-1).matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(37, 16, 8)
key = torch.randn(37, 6, 8)
value = torch.randn(37, 6, 8)
dropout_p = torch.nn.functional.dropout(torch.rand(1), p=0.5)
