
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1. / np.sqrt(14 * 14)
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, q, k, v, mask):
        dropout_qk = self.dropout(torch.matmul(q, k.transpose(-2, -1)).div(self.scale_factor))
        dropout_qk.masked_fill_(mask, float('-inf'))
        softmax_qk = torch.nn.functional.softmax(dropout_qk, dim=-1)
        dropout_softmax_qk = self.dropout(softmax_qk)
        return dropout_softmax_qk.matmul(v)

# Initializing the model inputs
q = torch.randn(2, 8, 14, 14)
k = torch.randn(2, 8, 14, 14)
v = torch.randn(2, 8, 14, 14)
mask = torch.zeros(2, 14, 14).bool()

# Initializing the model
m = Model()

# Inputs to the model
