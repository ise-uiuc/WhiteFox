
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout_qk = torch.nn.Dropout(dropout_p)
        self.attention_head_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, hidden_size)
 
    def forward(self, x1, x2):
        size = x2.shape[-1]
        scale_factor = np.sqrt(size / self.num_heads)
        proj1 = self.attention_head_projection(x1)
        proj2 = self.attention_head_projection(x2)
        qk = torch.matmul(proj1, proj2.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = torch.matmul(dropout_qk, x2)
        output = self.output(output)
        return output

# Initializing the model
m = Model(hidden_size=8, num_heads=2)

# Inputs to the model
x1 = torch.randn(1, 24, 8)
x2 = torch.randn(2, 24, 8)
