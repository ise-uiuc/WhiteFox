
class Model(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.fc_q = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_k = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_v = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        v = torch.nn.functional.dropout(value, self.dropout_p)
        scaled_qk = qk.div(v.size(-1))
        softmax_qk = scaled_qk.softmax(dim=-1)
        v = torch.nn.functional.dropout(v.transpose(-2, -1).softmax(dim=-1).transpose(-2, -1), self.dropout_p)
        return torch.matmul(softmax_qk, v)

    def gen_sample(self, batch_size):
        return torch.randint(0, self.hidden_size, size=(batch_size, 3, self.hidden_size)), torch.randn(batch_size, 3, self.hidden_size)

# Initializing the model
m = Model(512, 0.1)

# Inputs to the model
x1, x2 = m.gen_sample(1)
