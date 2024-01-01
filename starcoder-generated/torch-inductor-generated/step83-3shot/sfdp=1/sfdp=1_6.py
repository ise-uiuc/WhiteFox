s
class LitAttention1(LitModule):
    def __init__(self, dim_hidden: int, num_heads: int, scale_factor: int, dropout_p: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.query = nn.Linear(dim_hidden, dim_hidden)
        self.key = nn.Linear(dim_hidden, dim_hidden)
        self.value = nn.Linear(dim_hidden, dim_hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_p)
 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output


class LitAttention2(LitModule):
    def __init__(self, dim_hidden: int, num_heads: int, scale_factor: int, dropout_p: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.query = nn.Linear(dim_hidden, dim_hidden)
        self.key = nn.Linear(dim_hidden, dim_hidden)
        self.value = nn.Linear(dim_hidden, dim_hidden)
        self.dropout = nn.Dropout(p=dropout_p)
 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.query(q)
        q = q.view(q.size(0), q.size(1), num_heads, -1)
        q = q.permute(0, 2, 1, 3)

        k = self.key(k)
        k = k.view(k.size(0), k.size(1), num_heads, -1)
        k = k.permute(0, 2, 1, 3)
 
        v = self.value(v)
        v = v.view(v.size(0), v.size(1), num_heads, -1)
        v = v.permute(0, 2, 1, 3)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = nn.Softmax(dim=-1)(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)

        output = output.permute(0, 2, 1, 3)
        output = output.flatten(-2)
        return output


class LitAttention3(LitModule):
    def __init__(self, dim_hidden: int, num_heads: int, scale_factor: int, dropout_p: float):
        super().__init__()
        self.linear_query = nn.Linear(dim_hidden, dim_hidden)
        self.linear_key = nn.Linear(dim_hidden, dim_hidden)
        self.linear_value = nn.Linear(dim_hidden, dim_hidden)
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(p=dropout_p)
 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.linear_query(q)
        q = q.view(q.size(0), q.size(1), num_heads, -1)
        q = q.permute(0, 2, 1, 3)

        k = self.linear_key(k)
        k = k.view(k.size(0), k.size(1), num_heads, -1)
        k = k.permute(0, 2, 1, 3)
 
        v = self.linear_value(v)
        v = v.view(v.size(0), v.size(1), num_heads, -1)
        v = v.permute(0, 2, 1, 3)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = nn.Softmax(dim=-1)(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)

        output = output.permute(0, 2, 1, 3)
        output = output.flatten(-2)
        return output

# Initializing the model
m1, m2, m3 = LitAttention1(1024, 6, 1024 ** -0.5, 0.15), LitAttention2(1024, 6, 1024 ** -0.5, 0.15), LitAttention3(1024, 6, 1024 ** -0.5, 0.15)

# Inputs to the model
__m1_output__ = m1(x1, x2, x3)
__m2_output__ = m2(x1, x2, x3)
__m3_output__ = m3(x1, x2, x3)

