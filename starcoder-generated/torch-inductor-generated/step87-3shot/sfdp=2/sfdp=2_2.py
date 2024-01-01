
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout):
        super().__init__()
        self.wq = torch.nn.Linear(d_model, d_head)
        self.wk = torch.nn.Linear(d_model, d_head)
        self.wv = torch.nn.Linear(d_model, d_head)
        self.dropout_qk = torch.nn.Dropout(dropout)
        self.dropout_v = torch.nn.Dropout(dropout)
 
    def forward(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
 
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)).div(np.sqrt(k.size(-1)))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = self.dropout_v(dropout_qk).matmul(v)
        return output

# Initializing the model
m = Model(16, 256, 64, 0.1)

# Inputs to the model
q = torch.randn(4, 16, 256)
k = torch.randn(6, 16, 256)
v = torch.randn(5, 16, 256)
