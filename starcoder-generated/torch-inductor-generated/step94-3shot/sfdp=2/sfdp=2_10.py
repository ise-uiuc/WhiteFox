
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 8, 32, 32))
        self.key = torch.nn.Parameter(torch.randn(1, 8, 32, 32))
        self.value = torch.nn.Parameter(torch.randn(1, 8, 32, 32))
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout.matmul(v)
        return output

# Initializing the model
m = Model()
q = torch.randn(1, 8, 32, 32)
k = torch.randn(1, 8, 32, 32)
v = torch.randn(1, 8, 32, 32)
inv_scale_factor = 0.1
dropout_p = 0.5
