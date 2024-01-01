
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model for inference
query = torch.randn(1, 8, 16, 16)
key = torch.randn(1, 8, 32, 32)
value = torch.randn(1, 8, 32, 32)
inv_scale_factor = torch.randn(1, 1, 1, 1).div(8).clamp(min=1.0)
dropout_mask = torch.zeros(1, 8, 16, 16).bernoulli_(1.0 - dropout_p)
