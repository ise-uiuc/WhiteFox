
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p):
        super().__init__()
        self.qk = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False)
        self.dropout_p = dropout_p

    def forward(self, x1, x2):
        qk = self.qk(x1)
        qk_scaled = qk * 10
        softmax_qk = qk_scaled.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model(3, 4, 0.1)

# Inputs to the model
x1 = torch.randn(1, 3, 5, 5) # query
x2 = torch.randn(1, 4, 6, 6) # key
