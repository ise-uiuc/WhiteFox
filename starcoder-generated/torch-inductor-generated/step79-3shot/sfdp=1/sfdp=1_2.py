
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y, scale_factor):
        qk = torch.matmul(x, y.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 3, 64, 64)
scale_factor = 1e-5
