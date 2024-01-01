
class Model(torch.nn.Module):
    def __init__(self, in_dim, num_heads, ffn_hidden_dim, dropout_p, device):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout_p = dropout_p
        self.device = device
        self.head_dim = in_dim // num_heads
        self.scaled_in_dim = in_dim * num_heads
        self.fc_q = torch.nn.Linear(in_dim, self.scaled_in_dim, bias=True)
        self.fc_k = torch.nn.Linear(in_dim, self.scaled_in_dim, bias=True)
        self.fc_v = torch.nn.Linear(in_dim, self.scaled_in_dim, bias=True)
        self.fc_out = torch.nn.Linear(self.scaled_in_dim, in_dim, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, x1):
        q = self.fc_q(x1).view(x1.size(0), self.num_heads, self.in_dim // self.num_heads, x1.size(-1))
        k = self.fc_k(x1).view(x1.size(0), self.num_heads, self.in_dim // self.num_heads, x1.size(-1))
        v = self.fc_v(x1).view(x1.size(0), self.num_heads, self.in_dim // self.num_heads, x1.size(-1))
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / float(q.size(-1) ** 0.5)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        output = output.reshape(output.size(0), -1, self.in_dim)
        output = self.fc_out(output)
        return output

# Initializing the model
m = Model(20, 4, 25, 0.1, 'cpu')

# Inputs to the model
x1 = torch.randn(1, 10, 20)
