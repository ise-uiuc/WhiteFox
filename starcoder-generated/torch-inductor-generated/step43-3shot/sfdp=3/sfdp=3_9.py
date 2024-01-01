
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.6):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, x):
        n, c, h, w = x.shape
        q = torch.randn(n, c, 1, 1)
        k = torch.randn(n, c, h, w)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)).mul(1.0 / np.sqrt(c))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        o = torch.matmul(dropout_qk, x)
        return o
