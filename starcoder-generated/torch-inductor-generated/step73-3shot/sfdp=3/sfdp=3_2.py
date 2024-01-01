
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
scale_factor = torch.arange(1, 17).reshape(1, 1, 1, -1).float()[:, :, :, :8]
dropout_p = 0
query = torch.rand(1, 2, 8, 8)
key = torch.rand(1, 2, 8, 8)
value = torch.rand(1, 2, 8, 8)
