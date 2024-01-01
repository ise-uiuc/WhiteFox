
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, 8, 8))
        self.key = torch.nn.Parameter(torch.randn(1, 1, 16, 16))
        self.value = torch.nn.Parameter(torch.randn(1, 1, 32, 32))
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x):
        v1 = torch.matmul(x, self.query)
        v2 = torch.matmul(v1, self.key.transpose(-2, -1))
        v3 = v2.div(self.inv_scale_factor)
        v4 = torch.softmax(v3, dim=-1)
        