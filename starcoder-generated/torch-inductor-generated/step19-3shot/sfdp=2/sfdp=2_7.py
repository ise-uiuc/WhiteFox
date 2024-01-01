
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.q = torch.randn(8, 64, 32)
        self.k = torch.randn(8, 32, 64)
        self.v = torch.randn(8, 32, 64)
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
    
    def forward(self, query):
        qk = torch.matmul(query, self.k.transpose(-2, -1))
        inv_scale_factor = self.scale_factor ** -1
        dropout_qk = torch.nn.functional.dropout(qk.softmax(dim=-1) * inv_scale_factor, p=self.dropout_p)
        output = dropout_qk.matmul(self.v)
        return output

# Initializing the model
m = Model(0.75, 0.3)

# Inputs to the model
query = torch.randn(1, 8, 64, 32)
