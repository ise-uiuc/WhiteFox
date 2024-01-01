
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        scale_factor = 0.7
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = model(dropout_p=0.5)

# Inputs to the model
x1 = torch.randn(4, 8, 32, 32)
x2 = torch.randn(4, 8, 32, 32)
x3 = torch.randn(4, 8, 32, 32)
