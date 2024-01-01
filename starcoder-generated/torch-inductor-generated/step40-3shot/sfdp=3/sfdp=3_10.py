
class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(10)

# Inputs to the model
query = torch.randn(4, 2, 5, 10)
key = torch.randn(4, 2, 8, 10)
value = torch.randn(4, 2, 8, 5)
scale_factor = torch.tensor([1.0 / np.sqrt(5.0)])
dropout_p = 0.3
