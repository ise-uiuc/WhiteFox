
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = torch.nn.Parameter(torch.ones(1) * 0.9)
        self.scale_factor = torch.nn.Parameter(torch.ones(1))

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
Q = torch.randn(1, 16, 64, 64)
K = torch.randn(1, 16, 64, 64)
V = torch.randn(1, 16, 64, 64)
