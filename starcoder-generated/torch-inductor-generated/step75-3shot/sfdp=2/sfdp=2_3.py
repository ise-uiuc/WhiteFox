
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output[0]

# Initializing the model
inv_scale_factor = 2
dropout_p = 0.3
m = Model(inv_scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(1, 32, 512)
key = torch.randn(1, 64, 512)
value = torch.randn(1, 64, 512)
