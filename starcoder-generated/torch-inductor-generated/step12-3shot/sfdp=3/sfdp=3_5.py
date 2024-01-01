
class Model(torch.nn.Module):
    def __init__(self, hidden_size=128, nhead=8, scaling_factor=0.2):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.scaling_factor = scaling_factor
        self.fc_query = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_key = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_value = torch.nn.Linear(hidden_size, hidden_size)
        #self.fc_out = torch.nn.Linear(hidden_size, hidden_size)
        
    def forward(self, q, k, v, dropout_p):
        q = self.fc_query(q)
        k = self.fc_key(k)
        v = self.fc_value(v)
        
        q = q.view(-1, q.size(1), self.nhead, self.hidden_size // self.nhead).transpose(0,1) # [batch_size, seq_length, nhead, hidden_size // nhead]
        k = k.view(-1, k.size(1), self.nhead, self.hidden_size // self.nhead).transpose(0,1) # [batch_size, seq_length, nhead, hidden_size // nhead]
        v = v.view(-1, v.size(1), self.nhead, self.hidden_size // self.nhead).transpose(0,1) # [batch_size, seq_length, nhead, hidden_size // nhead]
        
        q = q / math.sqrt(self.hidden_size)
        _sq = torch.sum(q**2, -1) # [batch_size, seq_length, nhead]
        _sk = torch.matmul(k.transpose(-1,-2), q) / math.sqrt(self.hidden_size) # [batch_size, nhead, seq_length]
        
        _s = _sq[...,None] +  _sk[...,None] + (_sk.transpose(-1, -2)-_sq[...,None]).abs() * self.scaling_factor
        
        attn = torch.softmax(_s, dim=-1) # [batch_size, seq_length, nhead]
        if dropout_p > 0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        output = torch.matmul(attn, v) # [batch_size, seq_length, nhead, hidden_size // nhead]
        output = output.transpose(0, 1).contiguous().view(output.size()[1], -1) # [batch_size, seq_length, hidden_size]
        return output #(output + out*0)*0.5 + out*0.5
        
# Initializing the model
m = Model()

# Inputs to the model
q = torch.rand(3, 5, 128)
k = torch.rand(3, 6, 128)
v = torch.rand(3, 6, 128)
dropout_p = 0.2
out = m(q, k, v, dropout_p=dropout_p)

