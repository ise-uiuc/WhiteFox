
class QueryKeyAttentionModel(torch.nn.Module):
    def __init__(self, num_heads, scaling_factor=1. / math.sqrt(77), dropout_p=0.2):
        super().__init__()
        
        self.scaling_factor = scaling_factor
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.project_query = torch.nn.Linear(77, 22, bias=False)
        self.project_key = torch.nn.Linear(77, 22, bias=False)
        self.project_value = torch.nn.Linear(77, 22)

    def forward(self, query, key):
        q2 = self.project_query(query)
        k2 = self.project_key(key)
        v2 = self.project_value(key)
        
        q3 = q2.view(q2.size(0), q2.size(1), self.num_heads, 22 // self.num_heads)
        q4 = q3.permute(0, 2, 1, 3)
        k3 = k2.view(k2.size(0), k2.size(1), self.num_heads, 22 // self.num_heads)
        k4 = k3.permute(0, 2, 1, 3)
        
        qk = torch.matmul(q4, k4.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk * self.scaling_factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = torch.matmul(dropout_qk, v2) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = QueryKeyAttentionModel(num_heads = 2)

x1 = torch.randn(1, 120, 22)
x2 = torch.randn(1, 200, 22)
