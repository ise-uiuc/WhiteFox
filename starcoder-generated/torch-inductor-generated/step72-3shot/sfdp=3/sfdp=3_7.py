
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale=1.0, dropout_p=-1):
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale
        softmax_qk = F.softmax(qk, dim=-1)
        if dropout_p > 0.0:
            dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        else:
            dropout_qk = softmax_qk
        return torch.matmul(dropout_qk, v)
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 8, 16)
k = torch.randn(3, 8, 16)
v = torch.randn(3, 8, 16)
