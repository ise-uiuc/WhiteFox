
class Model(torch.nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.lin_q = torch.nn.Linear(dim_model, dim_model)
        self.lin_k = torch.nn.Linear(dim_model, dim_model)
        self.lin_v = torch.nn.Linear(dim_model, dim_model)
        self.fc = torch.nn.Linear(dim_model, dim_model)
 
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, value, key, query, mask):
        q = self.lin_q(query).reshape(query.size(0), -1, self.num_heads, self.dim_model // self.num_heads)
        k = self.lin_k(key).reshape(key.size(0), -1, self.num_heads, self.dim_model // self.num_heads)
        v = self.lin_v(value).reshape(value.size(0), -1, self.num_heads, self.dim_model // self.num_heads)
 
        q = q.permute(0, 2, 1, 3).contiguous().reshape(-1, key.size(1), self.dim_model // self.num_heads)
        k = k.permute(0, 2, 1, 3).contiguous().reshape(-1, key.size(1), self.dim_model // self.num_heads)
        v = v.permute(0, 2, 1, 3).contiguous().reshape(-1, value.size(1), self.dim_model // self.num_heads)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = self.dropout(qk)
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        if mask is not None:
            softmax_qk = softmax_qk.masked_fill(mask[:, None, None, :].bool(), float('-inf'))
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v)
 
        output = output.contiguous().reshape(value.size(0), self.num_heads, -1, self.dim_model // self.num_heads)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(value.size(0), -1, self.dim_model).contiguous()
        output = self.fc(output)
        return output

# Initializing the model
m = Model(128, 8)

# Inputs to the model
value = torch.randn(4, 100, 128)
key = torch.randn(4, 100, 128)
query = torch.randn(4, 100, 128)
mask = torch.rand(4, 100, 100) < 0.3
