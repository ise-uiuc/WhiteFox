
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_rate):
        super().__init__()
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = int(self.attention_head_size * num_attention_heads)
        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)
        self.scale_factor = torch.sqrt(torch.tensor([self.attention_head_size]) * torch.tensor([hidden_size]))
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, v1, v2):
        q1 = self.query(v1)
        k1 = self.key(v2)
        v1 = self.value(v2)
        q2 = q1.view(q1.shape[0], q1.shape[1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        k2 = k1.view(k1.shape[0], k1.shape[1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        v2 = v1.view(v1.shape[0], v1.shape[1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        scaled_qk = torch.matmul(q2, k2.transpose(2, 3))
        scaled_qk = scaled_qk / self.scale_factor
        scaled_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(scaled_qk)
        output = dropout_qk.matmul(v2)
        return output

# Inputs to the model
hidden_size = 16 
num_attention_heads = 2
dropout_rate = 0.1 
v1 = torch.randn(1, 16, 64, 64)
v2 = torch.randn(1, 16, 64, 64)
