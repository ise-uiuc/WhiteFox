
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 64
        self.num_heads = 8
        self.head_dim = 64 // self.num_heads
        self.scale_factor = self.head_dim**-0.5
        self.q_linear = torch.nn.Linear(self.in_features, self.in_features)
        self.k_linear = torch.nn.Linear(self.in_features, self.in_features)
        self.v_linear = torch.nn.Linear(self.in_features, self.in_features)
        self.dropout = torch.nn.Dropout(0.01)
 
    def forward(self, query, key, value):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        output = torch.flatten(output, start_dim=1, end_dim=2)
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 32)
key = torch.randn(1, 64, 16)
value = torch.randn(1, 64, 16)
