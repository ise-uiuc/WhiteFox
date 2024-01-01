
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(20, 1, 512)
key = torch.randn(20, 10, 512)
value = torch.randn(20, 10, 64)
inv_scale_factor = torch.ones(1)
dropout_p = 0.2
