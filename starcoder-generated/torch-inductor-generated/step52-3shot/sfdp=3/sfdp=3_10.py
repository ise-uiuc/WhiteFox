
class Model(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, scale_factor=None, dropout_p=None, x3=None):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        if scale_factor is not None:
            scaled_qk = qk.mul(scale_factor)
        else:
            scaled_qk = qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        if dropout_p is not None:
            dropout_qk = nn.functional.dropout(softmax_qk, p=dropout_p)
        else:
            dropout_qk = softmax_qk
        if x3 is not None:
            output = dropout_qk.matmul(x3)
        else:
            output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 3)
x2 = torch.randn(1, 5, 4)
x3 = torch.randn(1, 4, 5)
