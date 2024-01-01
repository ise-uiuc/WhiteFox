
class Model(torch.nn.Module):
    def __init__(self, dim_q=128, dim_v=128, num_head=4, dropout_p=0.5, scale_factor=1.0 / np.sqrt(128)):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
        self.num_head = num_head
        
        dim = dim_q * num_head
        
        self.query = torch.nn.Linear(dim_q, dim, bias=False)
        self.key = torch.nn.Linear(dim_q, dim, bias=False)
        self.value = torch.nn.Linear(dim_v, dim, bias=False)
        
    def forward(self, x1, x2):
        q1 = self.query(x1)
        k1 = self.key(x1)
        v1 = self.value(x2)
        
        q2 = q1.reshape(q1.size(0), q1.size(1), self.num_head, -1)
        k2 = k1.reshape(k1.size(0), k1.size(1), self.num_head, -1)
        v2 = v1.reshape(v1.size(0), v1.size(1), self.num_head, -1)
        
        q3 = q2.transpose(-2, -1)
        k3 = k2.transpose(-2, -1)
        v3 = v2.transpose(-2, -1)
        
        q4 = torch.matmul(q3, k3)
        q5 = q4 * self.scale_factor
        q6 = torch.nn.functional.softmax(q5, dim=-1)
        q7 = torch.nn.functional.dropout(q6, self.dropout_p, True)
        q8 = torch.matmul(q7, v3)
        
        q9 = q8.reshape(q1.size(0), -1)
        q10 = q9.transpose(1, 0)
        
        return q10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128)
