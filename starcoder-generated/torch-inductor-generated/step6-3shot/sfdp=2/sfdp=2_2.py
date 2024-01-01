
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.rand(64, 10, 1))

    def forward(self, query, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing key and values
self.key = torch.nn.Parameter(torch.rand(64, 10, 1))
values = torch.rand(64, 10, 20)

# Inputs to the model
query = torch.randn(64, 10, 1)
inv_scale_factor = torch.uniform(1, 2)
dropout_p = torch.uniform(0, 1)
