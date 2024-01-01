
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(1, 128, 512)
key = torch.randn(1, 128, 512)
value = torch.randn(1, 128, 512)
scale_factor = torch.randn(1, 128, 1)
dropout_p = 0.3
