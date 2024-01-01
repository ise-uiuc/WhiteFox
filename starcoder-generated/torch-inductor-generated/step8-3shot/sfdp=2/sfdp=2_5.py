
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, inv_scale_factor, dropout_p):
        super().__init__()
        self.qk_w = torch.nn.Linear(dim, dim * num_heads, bias=True)
        self.linear_v = torch.nn.Linear(dim, dim * num_heads, bias=True)
        self.linear_o = torch.nn.Linear(dim * num_heads, dim, bias=True)
        self.softmax = torch.nn.Softmax(dim=3)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(2, 3))
        qk_w = self.qk_w(qk)
        qk_w_shape = qk_w.shape
        qk_w = qk_w.view(qk_w_shape + (1,1))[...,:,0].squeeze(-1)
        qk = torch.div(qk_w, self.inv_scale_factor)
        v = self.linear_v(v)
        v_shape = v.shape
        v = v.view(v_shape + (1,1))[...,:,0].squeeze(-1)
        qk = self.softmax(qk)
        qk = self.dropout(qk)
        qk = qk.unsqueeze(-1)
        output = torch.matmul(qk, v)
        output = self.linear_o(output)
        output = output + q
        return output

# Initializing the model
m = Model(dim=512, num_heads=16, inv_scale_factor=0.0625*(dim**-0.5), dropout_p=0.2)

# Inputs to the model
query = torch.randn(2, 6, 512)
key = torch.randn(2, 25, 512)
value = torch.randn(2, 25, 512)
