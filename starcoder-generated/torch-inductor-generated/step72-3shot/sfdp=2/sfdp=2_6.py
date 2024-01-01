
class Model(torch.nn.Module):
    def __init__(self, num_keys):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(num_keys, 64, 64))
        self.inv_scale_factor = torch.nn.Parameter(torch.randn(num_keys))
 
    def forward(self, x1, dropout_p=0.0):
        __v1__ = torch.matmul(x1, self.key.transpose(-2, -1))
        scaled_qk = __v1__.div(self.inv_scale_factor)
        v2 = scaled_qk.softmax(dim=-1)
        dropout_v2 = torch.nn.functional.dropout(v2, p=dropout_p)
        v3 = dropout_v2.matmul(self.key)
        return v3

# Initializing the model
m = Model(128)

# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
dropout_p = 0.0
