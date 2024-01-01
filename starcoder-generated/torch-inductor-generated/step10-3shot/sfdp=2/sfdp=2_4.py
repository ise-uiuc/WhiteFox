
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 6, 64, 64)
v = torch.randn(1, 6, 64, 64)
inv_scale_factor = 2.0
dropout_p = 0.2
