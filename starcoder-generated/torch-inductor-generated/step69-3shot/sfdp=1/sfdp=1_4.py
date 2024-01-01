
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, dropout_p):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = x1.size(-1) ** -0.25
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 1, 5, 5)
x2 = torch.randn(1, 1, 3, 3)
dropout_p=0.5
