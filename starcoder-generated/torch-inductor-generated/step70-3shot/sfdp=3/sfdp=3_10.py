
class Model(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, bias=True):
        super().__init__()
 
        self.num_heads = num_heads
        self.d_model = d_model
 
        assert d_model % num_heads == 0
 
        self.scale_factor = d_model ** -0.5
 
        self.w_q = torch.nn.Linear(d_model, d_model, bias)
        self.w_k = torch.nn.Linear(d_model, d_model, bias)
        self.w_v = torch.nn.Linear(d_model, d_model, bias)
        self.w_o = torch.nn.Linear(d_model, d_model, bias)
 
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x1, x2, x3):
        q = self.w_q(x1)
        k = self.w_k(x2)
        v = self.w_v(x3)
 
        q = q.view(q.size(0), q.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        q = q.contiguous().view(q.size(0) * q.size(1), q.size(2), q.size(3))
 
        k = k.view(k.size(0), k.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k = k.contiguous().view(k.size(0) * k.size(1), k.size(2), k.size(3))
 
        v = v.view(v.size(0), v.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        v = v.contiguous().view(v.size(0) * v.size(1), v.size(2), v.size(3))
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
 
        output = dropout_qk.matmul(v)
 
        output = (
            output.view(q.size(0), q.size(1), output.size(1), output.size(2))
          .transpose(1, 2)
          .contiguous()
          .view(output.size(0), output.size(2), output.size(1) * output.size(3))
        )
 
        return self.w_o(output)

# Instantiate the model with the given inputs.
num_heads = torch.randint(1, 8, (1,)).item()
d_model = torch.randint(16, 128, (1,)).item()
dropout = torch.rand(1,).item() * 0.4 + 0.1
bias = torch.randint(0, 2, (1,)).item() == 1
m = Model(num_heads, d_model, dropout, bias)

# Inputs to the model
x1 = torch.randn(1234, 16)
x2 = torch.randn(1234, 16)
x3 = torch.randn(2345, 16)
