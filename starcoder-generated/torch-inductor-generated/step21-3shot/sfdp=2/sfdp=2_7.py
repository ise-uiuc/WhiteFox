
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(4, 12)
        self.dropout = torch.nn.Dropout(0.5)
        
        # Add the inverse scale factor
        self.lin2 = torch.nn.Linear(4, 12)
        self.lin2.bias.data = -1 * torch.tensor([self.lin2.bias], dtype=torch.float64)
    
    def forward(self, query, key, value, dropout_p=0.5):
        k1 = self.lin1(key)
        q1 = self.lin1(query)
        k1 = self.dropout(k1)
        q1 = self.dropout(q1)
        
        k = torch.bmm(k1, key.permute(0, 2, 1).contiguous())
        k = k / k.max()

        q = torch.bmm(q1, query.permute(0, 2, 1).contiguous())
        q = q / q.max()
        
        dmk = torch.matmul(query, key.permute(0, 2, 1).contiguous())
        dmk = dmk / dmk.max()
        
        qk = torch.softmax(dmk, dim=-1)
        
        output = qk * value
        
        v1 = torch.matmul(output, key.permute(0, 2, 1).contiguous())
        v2 = v1 * 0.5
        v3 = v1 * 0.707106781187
        v4 = torch.erf(v1)
        v5 = v4 + 1
        v6 = v2 * v5
        
        return v6

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 4)
key = torch.randn(1, 12, 4)
value = torch.randn(1, 12, 4)
