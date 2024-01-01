
class Model(torch.nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dim = dim
        self.dropout = torch.nn.Dropout2d(dropout)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.FloatTensor(self.dim))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model(dim=34, dropout=0.1)

# Inputs to the model
x1 = torch.randn(56, 34, 24)
x2 = torch.randn(29, 34, 25)
