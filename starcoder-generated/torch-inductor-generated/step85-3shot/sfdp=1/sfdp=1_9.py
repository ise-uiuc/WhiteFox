
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 32768.0
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(x3)
        return v6



# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)
x2 = torch.randn(1, 8, 16, 20)
x3 = torch.randn(1, 8, 20, 32)
