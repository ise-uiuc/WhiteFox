
class Model():
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return torch.matmul(dropout_qk, value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 20, 100)
key = torch.randn(1, 20, 120)
value = torch.randn(1, 20, 120)
inv_scale_factor = 1.0E-6
dropout_p = 0.2
