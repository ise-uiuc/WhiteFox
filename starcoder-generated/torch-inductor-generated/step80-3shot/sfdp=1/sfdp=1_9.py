
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        return torch.matmul(scaled_qk.softmax(dim=-1), value)

#  Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64)
key = torch.randn(1, 3, 64)
value = torch.randn(1, 3, 64)
inv_scale_factor = 1.0
dropout_p = 0.5
