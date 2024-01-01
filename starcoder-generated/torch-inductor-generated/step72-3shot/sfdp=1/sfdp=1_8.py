
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        qk_inv_scale_factor = 1.0 / math.sqrt(x1.size(-1))
        qk_dropout_p = self.dropout_p
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=qk_dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64)
x2 = torch.randn(1, 16, 32)
