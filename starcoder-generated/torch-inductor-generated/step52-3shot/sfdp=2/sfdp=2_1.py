
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # This layer is required by the pattern
        self.matmul = torch.nn.MatMul(None, True)
 
    def forward(self, q1, k1, v1, inver_scale_factor, dropout_scale_factor):
        v1 = self.matmul(q1, k1.transpose(-2, -1))
        v2 = v1.div(inver_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = F.dropout(v3, dropout_p, False)
        dropout_qk = self.matmul(v4, v1)
        output = self.matmul(dropout_qk, v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 1, 8)
k1 = torch.randn(1, 8, 8)
