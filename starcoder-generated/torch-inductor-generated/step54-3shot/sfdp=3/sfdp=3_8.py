
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v1, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 3, 64, 64)
v1 = torch.randn(1, 3, 64, 64)
scale_factor = torch.randn(1, 64, 1)
dropout_p = 0.34951456358861597
