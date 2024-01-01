
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 2
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.5
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(softmax_qk)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 128)
x2 = torch.randn(1, 128, 64)
