
class Model(torch.nn.Module):
    def __init__(self, d_model=64, nhead=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.d_k = dim_feedforward // nhead
        self.linear1 = nn.Linear()
        self.dropout = nn.Dropout(p=dropout)
 
    def forward(self, q, q_lengths, k, v, v_lengths):
        q_mask = q.requires_grad_(False).to(torch.bool)[0, :, :]
        k_mask = k.requires_grad_(False).to(torch.bool)[0, :, :]
        lengths = [q_lengths, v_lengths]
        #...
     
        #...
        q_trans = q.transpose(0, 1)
        q_norm = torch.norm(q_trans, dim=-1, keepdim=True)  
        k_norm = torch.norm(k_trans, dim=-1, keepdim=True)
        a = k_trans.matmul(q_norm).div(k_norm) 
        a = torch.softmax(a, 0) 
        result = a.matmul(k) 
 
# Initializing the model
m = Model()

# Inputs to the model
d_model = 64
nhead = 2
m.d_model = d_model
m.nhead = nhead
x1 = torch.randn(2, 50, d_model)
x2 = torch.randn(2, 50, d_model)
q_lengths = torch.tensor([37, 12])
v_lengths = torch.tensor([51, 56])
