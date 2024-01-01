
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        q11_q44 = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1 / 127.0
        scaled_qk = q11_q44.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        o11_o33 = dropout_qk.matmul(x3)
        r11_r33 = dropout_qk.matmul(x4)
        return o11_o33, r11_r33

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 2)
x2 = torch.randn(2, 2, 4)
x3 = torch.randn(2, 3, 3)
x4 = torch.randn(2, 3, 3)
__output1__, __output2__ = m(x1, x2, x3, x4)