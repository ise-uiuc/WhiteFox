
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0., scale_factor=0.2):
        super().__init__()
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# initialize the model
x1, x2 = torch.randn(1, 64, 256), torch.randn(1, 256, 512)
dropout_p = 0.2
m = Model(dropout_p)
