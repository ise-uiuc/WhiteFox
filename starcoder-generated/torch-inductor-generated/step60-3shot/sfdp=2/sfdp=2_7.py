
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.key_proj = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.value_proj = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qv = torch.matmul(self.query_proj(query), self.key_proj(key).transpose(-2, -1))
        scaled_qk = qv.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = self.value_proj(dropout_qk.matmul(value))
        return qk


# Initializing the model
inv_scale_factor = 64.0
dropout_p = 0.1

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(10, 3, 64, 64)
value = torch.randn(10, 3, 64, 64)

