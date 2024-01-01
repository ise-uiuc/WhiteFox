
class Attention(torch.nn.Module):
    def __init__(self, h_size, dropout):
        super(Attention, self).__init__()
        scale = 1/math.sqrt(h_size)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(h_size, h_size, bias=False)
        self.linear2 = torch.nn.Linear(h_size, 1, bias=False)
        self.scale = torch.nn.Parameter(torch.FloatTensor([scale]).reshape(1, 1, 1))
        
    def forward(self, query, key, value, mask=None):
        scale_factor = self.scale.expand_as(query)
        qk = torch.matmul(query, key.transpose(-1, -2))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output


# Initializing the model
torch.manual_seed(1643)
h_size, dropout, seq_length = 64, 0.2, 32
q, k, v = torch.randn(1, seq_length, h_size), torch.randn(1, seq_length, h_size), torch.randn(1, seq_length, h_size)
