
class Model(torch.nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        head_dim = dim_model // num_heads
        self.linear_query = torch.nn.Linear(dim_model, dim_model)
        self.linear_key = torch.nn.Linear(dim_model, dim_model)
        self.linear_value = torch.nn.Linear(dim_model, dim_model)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v, mask, inv_scale_factor):
        q = self.linear_query(q).view(q.size(0), q.size(1), q.size(2), num_heads, head_dim).transpose(2, 3)
        k = self.linear_key(k).view(k.size(0), k.size(1), k.size(2), num_heads, head_dim).transpose(2, 3)
        v = self.linear_value(v).view(v.size(0), v.size(1), v.size(2), num_heads, head_dim).transpose(2, 3)
        q = q.div(math.sqrt(head_dim))
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = scaled_qk.div(inv_scale_factor)
        softmax_q = self.softmax(scaled_qk.float())
        softmax_q = softmax_q.to(dtype=q.dtype)
        dropout_q = self.dropout(softmax_q)
        output = torch.matmul(dropout_q, v)
        output = output.view(output.size(0), output.size(1), output.size(2), -1).transpose(2, 3)
        output = output.contiguous().view(output.size(0), output.size(1), -1)
        return output

# Initializing the model
m = Model(dim_model=256, num_heads=16, dropout_p=0.12)

# Inputs to the model
q = torch.randn(2, 20, 1, 1, 256)
k = torch.randn(2, 20, 1, 1, 256)
v = torch.randn(2, 20, 1, 1, 256)
mask = torch.ones(q.size(0), q.size(1), k.size(1)).to(dtype=torch.bool).to(q.device)
inv_scale_factor = torch.ones((1, 1, 1, 1)).to(q.device)
output = m(q, k, v, mask, inv_scale_factor)

