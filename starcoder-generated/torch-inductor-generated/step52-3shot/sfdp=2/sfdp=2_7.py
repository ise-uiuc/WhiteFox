
class Model(nn.Module):
    def __init__(self, num_query, dim_query, num_key, dim_key, num_value, dim_value, inv_scale_factor, dropout_p, use_bias):
        super().__init__()
        self.num_query = num_query
        self.dim_query = dim_query
        self.num_key = num_key
        self.dim_key = dim_key
        self.num_value = num_value
        self.dim_value = dim_value
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
        self.use_bias = use_bias
        self.scale = 1 / (self.dim_value ** 0.5)
        self.linear_query = nn.Linear(self.num_query, self.num_key, bias=self.use_bias)
        if self.linear_query.bias is not None:
            self.linear_query.bias.data.uniform_(-0.1, 0.1)
        self.linear_key = nn.Linear(self.num_key, self.dim_key, bias=self.use_bias)
        if self.linear_key.bias is not None:
            self.linear_key.bias.data.uniform_(-0.1, 0.1)
        self.linear_value = nn.Linear(self.num_value, self.dim_value, bias=self.use_bias)
        if self.linear_value.bias is not None:
            self.linear_value.bias.data.uniform_(-0.1, 0.1)
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(self.dropout_p)
 
    def forward(self, x1, x2, x3):
        q_seq = x1.reshape(-1, self.num_query, self.num_key)
        query = self.linear_query(q_seq).unsqueeze(dim=-2)
        k_seq = x2.reshape(-1, self.num_key, self.num_value)
        key = self.linear_key(k_seq).transpose(1, 2)
        v_seq = x3.reshape(-1, self.num_value, self.dim_value)
        value = self.linear_value(v_seq).transpose(1, 2)
        qk = torch.matmul(query, key)
        scaled_qk = qk * self.scale * self.inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output.view(x1.shape[0], x1.shape[1], self.dim_value)

# Initializing the model
m = Model(2, 4, 3, 6, 4, 10, 1.0, 0.3, True)

# Inputs of the model
x1 = torch.randn(1, 2, 4)
x2 = torch.randn(1, 3, 6)
x3 = torch.randn(1, 4, 10)

# Run the model
