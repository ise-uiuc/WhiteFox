


class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, d_model, num_head):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.d_model = d_model
        self.num_head = num_head 
        self.query_proj = torch.nn.Linear(query_size, d_model * num_head)
        self.key_proj = torch.nn.Linear(key_size, d_model * num_head)
        self.value_proj = torch.nn.Linear(value_size, d_model * num_head)
        self.out_proj = torch.nn.Linear(d_model * num_head, d_model)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor):
        q = self.query_proj(query).view(-1, self.num_head, self.d_model)
        k = self.key_proj(key).view(-1, self.num_head, self.d_model)
        v = self.value_proj(value).view(-1, self.num_head, self.d_model)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
  
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(-1, self.num_head * self.d_model)
        return self.out_proj(output)

# Initializing the model
m = Model(input_shapes['query'], input_shapes['key'], input_shapes['value'], args.d_model, args.num_heads)

# Inputs to the model
query_value = torch.randn(1, 532, 128)
inv_scale_factor = torch.rand(1, 10, 10, 10)
