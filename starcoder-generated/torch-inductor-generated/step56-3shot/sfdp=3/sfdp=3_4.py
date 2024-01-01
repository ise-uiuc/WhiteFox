
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(query, key, value, scale_factor, dropout_p):
        matmul = torch.matmul(query, key.transpose(-2, -1))
        scale = matmul.mul(scale_factor)
        softmax_qk = scale.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        result = dropout.matmul(value)
        return result

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 40)
key = torch.randn(1, 12, 40)
value = torch.randn(1, 12, 40)
scale_factor = torch.tensor(50)
dropout_p = torch.tensor(0.1)
