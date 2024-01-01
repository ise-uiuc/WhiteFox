
class MySelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p, device=None):
        super(MySelfAttention, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.hidden_size = hidden_size
        self.head_num = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout_p
        self.scale_factor = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout_p)
        self.to(device)

    def forward(self, input_tensor):
        query_tensor = self.W_q(input_tensor)
        key_tensor = self.W_k(input_tensor)
        value_tensor = self.W_v(input_tensor)
        qk_tensor = torch.matmul(query_tensor, key_tensor.transpose(-2, -1)) / self.scale_factor
        softmax_qk_tensor = qk_tensor.softmax(dim=-1)
        dropout_qk_tensor = self.dropout(softmax_qk_tensor)
        output_tensor = torch.matmul(dropout_qk_tensor, value_tensor)
        return output_tensor

m = MySelfAttention(hidden_size=160, num_heads=2, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(96, 160)
