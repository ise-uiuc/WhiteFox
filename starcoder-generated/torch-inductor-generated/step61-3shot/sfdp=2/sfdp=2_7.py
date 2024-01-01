
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.2, dim=512, num_heads=4, window_size=512):
        super().__init__()
        self.scale_factor = (dim//num_heads)**(-0.5)
        self.inv_scale_factor = (dim//num_heads)**(0.5)
        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.window_size = window_size
 
    def forward(self, query, key, value):
        q = torch.nn.functional.dropout(query, p=dropout_p)
        q = self.linear_q(q)
        k = torch.nn.functional.dropout(key, p=dropout_p)
        k = self.linear_k(k)
        v = torch.nn.functional.dropout(value, p=dropout_p)
 
        q = q.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1).transpose_(1, 2)
        k = k.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1).transpose_(1, 2)
 
        q = self.wq(q).transpose(1, 2).transpose(2, 3)
        k = self.wk(k).transpose(1, 2).transpose(2, 3)
 
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
 
        unsqueezed = False
        if q.dim() == 3:
            q = q.unsqueeze(1)
            unsqueezed = True
        elif k.dim() == 3:
            k = k.unsqueeze(1)
            unsqueezed = True
 
        qk = torch.matmul(q, k)
 
        unfolded = False
        if qk.dim() == 3:
            qk = qk.unsqueeze(0)
            unfolded = True
        else:
            qk = qk.unsqueeze(1).unsqueeze(-1)
        
        if unsqueezed:
            qk = qk.squeeze(1)
 
        qk = qk.mul(1/self.inv_scale_factor)
 
        output = qk.softmax(dim=-1)
        output = torch.nn.functional.dropout(output, p=dropout_p)
        output = torch.matmul(output, v).squeeze(-1).unsqueeze(-1)
 
        if unsqueezed:
            output = output.squeeze(1)
        if unfolded:
            output = output.squeeze(0)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 32, 32)
key = torch.randn(1, 4, 32, 32)
value = torch.randn(1, 4, 32, 32)
