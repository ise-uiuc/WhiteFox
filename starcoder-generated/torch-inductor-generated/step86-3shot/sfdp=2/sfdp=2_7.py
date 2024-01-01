
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.drop     = torch.nn.Dropout(p=0.5)
 
    def forward(self, q, k, v, dropout=True):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = np.sqrt(q.size(-1))
        scaled_qk      = qk / inv_scale_factor
        softmax_qk     = self.softmax(scaled_qk)
        if dropout is True:
            dropout_qk = self.drop(softmax_qk)
        else:
            dropout_qk = softmax_qk
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 3, 64, 64)
k = torch.randn(2, 3, 64, 64)
v = torch.randn(2, 3, 64, 64)
