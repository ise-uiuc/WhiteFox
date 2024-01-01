
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, dropout_p):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(x1.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 512)
x2 = torch.randn(1, 128, 128)
dropout_p = torch.rand(()).cpu()
