
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout = dropout_p
 
    def forward(self, q, k, v, dropout_p):
        dk = torch.tensor(k.shape[-1]).float()
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = ((dk // 2) ** -0.5)
        qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(0.1)

# Inputs to the model
q = torch.randn(2, 8, 512)
k = torch.randn(2, 8, 512)
v = torch.randn(2, 8, 512)
dropout_p = torch.tensor(0.1)
