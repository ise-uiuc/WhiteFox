
class Model(torch.nn.Module):

    def __init__(self, heads, hidden_dim):
        super().__init__()
        self.scale_factor = (hidden_dim/heads)**-0.5
 
        self.query = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.query.weight, gain=1.414))
        self.key = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.key.weight, gain=1.414))
        self.value = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.value.weight, gain=1.414))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, x):
        v1 = self.query(x)  # N x l h
        v2 = self.key(x)    # N x l h
        v3 = self.dropout(self.softmax(self.scale_factor * torch.matmul(v1, v2.transpose(-2, -1)))) # N x l h
        v4 = self.value(x)   # N x l h
        output = torch.matmul(v3, v4)  # N x l h
        return output
        
# Initializing the model
m = Model(32, 1024)

# Inputs to the model
x1 = torch.randn(128, 3072)
