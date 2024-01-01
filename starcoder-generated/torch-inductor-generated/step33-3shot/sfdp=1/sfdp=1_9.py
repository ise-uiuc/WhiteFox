
class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_p = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scaled_div = 1/sqrt(d_model)
 
    def forward(self, q, k, v):
        _q = self.w_q(q).view(q.size(0), self.h, self.d_k, q.size(1))
        _k = self.w_k(k).view(k.size(0), self.h, self.d_k, k.size(1))
        _v = self.w_v(v).view(v.size(0), self.h, self.d_k, v.size(1))
        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)
        _qk = torch.matmul(_q, _k.transpose(-2, -1))
        _qk *= self.scaled_div
        _dropout_q = torch.nn.functional.dropout(torch.nn.functional.softmax(_qk, dim=-1), p=0.1)
        _output = _dropout_q.matmul(_v).transpose(1, 2).contiguous().view(q.size(0), -1, self.h*self.d_k)
        _output = self.fc(_output)
        return _output

# Initializing the model
m = MultiheadAttention(d_model=16, num_heads=2, dropout_p=0.2)

# Inputs to the model
x = torch.randn(2, 6, 16)
y = torch.randn(2, 5, 16)
z = torch.randn(2, 5, 16)
