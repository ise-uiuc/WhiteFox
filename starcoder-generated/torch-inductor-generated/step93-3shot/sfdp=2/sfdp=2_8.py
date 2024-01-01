
class Model(torch.nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.q = torch.nn.Linear(20, 300)
        self.k = torch.nn.Linear(20, 300)
        self.v = torch.nn.Linear(20, 300)
        self.dropout = torch.nn.Dropout(0.1)
        self.norm_factor = torch.sqrt(torch.FloatTensor([dim_size])).to(device)
        self.scale_factor = torch.FloatTensor([dim_size ** -0.5]).to(device)
 
    def forward(self, query, key, value):
        q = torch.nn.functional.relu(self.q(query))
        k = torch.nn.functional.relu(self.k(key))
        v = torch.nn.functional.relu(self.v(value))
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = self.dropout(dropout_qk).matmul(v)
        return output

# Initializing the model
m = Model(dim_size=20)

# Inputs to the model
query = torch.randn(20, 20)
key = torch.randn(20, 20)
value = torch.randn(20, 20)
