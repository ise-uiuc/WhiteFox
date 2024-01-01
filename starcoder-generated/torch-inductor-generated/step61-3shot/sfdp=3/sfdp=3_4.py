
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        q = x1
        k = x2
        v = x3
        scale_factor = self.scale_factor or (int(q.size(-1)) ** -0.5)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 768, 56, 56)
key = torch.randn(2, 768, 28, 28)
value = torch.randn(2, 768, 28, 28)
