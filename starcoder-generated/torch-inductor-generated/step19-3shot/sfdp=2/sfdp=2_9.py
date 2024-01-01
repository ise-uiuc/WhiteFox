
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        output = scaled_qk.softmax(dim=-1).matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 12, 512)
key = torch.randn(5, 12, 512)
value = torch.randn(5, 12, 512)
