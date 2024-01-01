
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p = 0.):
        qk = torch.matmul(query, key.transpose(-2, -1))
        weighted_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.Softmax(dim=-1)(weighted_qk)
        dropout_qk = torch.nn.Droupout(dropout_p)(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
