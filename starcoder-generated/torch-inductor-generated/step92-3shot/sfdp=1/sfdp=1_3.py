
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0
        self.scale_factor = 0.0
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3, x4):
        q = self.conv(x1)
        k = self.conv(x2)
        v = self.conv(x3)
        inv_scale_factor = 1 / self.scale_factor
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 32)
x2 = torch.randn(1, 3, 32, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 32, 32)
