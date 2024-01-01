
class Model(torch.nn.Module):
    def __init__(self, batch, num, dims, dropout):
        super().__init__()
        self.batch = batch
        self.num = num
        self.dims = dims
        self.dropout = dropout
        self.dropout_p = dropout / (dims * num)
        
        self.qk = torch.nn.Linear(dims, dims, bias=False)
        self.v = torch.nn.Linear(dims, dims, bias=False)

        self.dropout_qk = torch.nn.Dropout(p=self.dropout_p)
        
    def forward(self, q, k, v, mask=None):
        qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_qk = qk.div(self.batch**0.5)
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == 0, -1e10)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        o = torch.matmul(dropout_qk, v)
        return o

# Initializing the model
batch = 8
num = 6
dims = 512
dropout = 0.1
m = Model(batch=batch, num=num, dims=dims, dropout=dropout)

# Inputs to the model
q = torch.randn(batch, num, dims)
k = torch.randn(batch, num, dims)
v = torch.randn(batch, num, dims)
mask = torch.randint(2, (batch, num))
