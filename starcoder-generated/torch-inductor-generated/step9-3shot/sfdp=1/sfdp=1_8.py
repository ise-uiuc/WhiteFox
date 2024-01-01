
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.value = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        k = self.key(x1)
        v = self.value(x1)
        q = torch.randn(1, 4, 64, 64)
        __key__ = k
        __value__ = v
        __query__ = q
        qk = torch.matmul(__query__, __key__.transpose(-2, -1))
        inv_scale_factor = 1. / 4
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.1
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(__value__)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
