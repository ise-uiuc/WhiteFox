
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 4)
 
    def forward(self, x1, x2):
        q = torch.cat([x1, x2], dim=-1)
        k = self.fc(q)
        inv_scale_factor = 1
        dropout_p = 0
        v = torch.cat([x1, x2], dim=-1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
