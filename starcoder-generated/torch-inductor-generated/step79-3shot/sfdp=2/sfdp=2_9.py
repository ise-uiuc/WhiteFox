
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(32, 8)
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, q1, k1, v1):
        q2 = self.dense(q1)
        k2 = self.dense(k1)
        v2 = self.dense(v1)
        q3 = torch.matmul(q2, k2.transpose(-2, -1))
        v3 = torch.matmul(q2, v2.transpose(-2, -1))
        scale_factor = torch.mean(q3)
        inv_scale_factor = 1 / scale_factor
        q4 = q3 * inv_scale_factor
        softmax = q4.softmax(dim=-1)
        dropout = self.dropout(softmax)
        output = dropout.matmul(v3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 8, 4)
k1 = torch.randn(1, 16, 4)
v1 = torch.randn(1, 16, 4)
