
class Model(torch.nn.Module):
    def __init__(self, p=0.5, inv_scale_factor=10):
        super().__init__()
        self.dropout = torch.nn.Dropout(p)
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, x3)
        return output

# Initializing the model
p = 0.4
inv_scale_factor = 2
m = Model(p, inv_scale_factor)

# Inputs to the model
x1 = torch.randn(128, 8, 64)
x2 = torch.randn(128, 64, 128)
x3 = torch.randn(128, 128, 64)
