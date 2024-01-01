
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(np.power(x1.shape[-1], 2))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model1()

# Inputs to the model
p = 0.38438
x1 = torch.randn(1, 5, 3)
x2 = torch.randn(3, 5, 7)
x3 = torch.randn(1, 5, 7)
