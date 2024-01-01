
class Model(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dim5, dropout_p=0.7, scale_factor=2048**(-1/4)):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk * 2048**(-1/4)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, x3)
        return output

# Initializing the model
m = Model(dim1=32, dim2=128)

# Inputs to the model
x1 = torch.randn(1, 8, 32, 64)
x2 = torch.randn(1, 8, 128, 64)
x3 = torch.randn(1, 128, 64, 64)

