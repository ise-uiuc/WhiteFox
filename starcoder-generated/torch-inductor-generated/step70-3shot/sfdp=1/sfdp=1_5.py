
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1, bias1, noise1):
        q = torch.nn.functional.normalize(q1, dim=-1, p=2)
        k = torch.nn.functional.normalize(k1, dim=-1, p=2)
        v = torch.nn.functional.normalize(v1, dim=-1, p=2)
        i = torch.bmm(q, k.transpose(-2, -1))
        attn = torch.bmm(self.dropout(torch.nn.functional.softmax(i + bias1.unsqueeze(0) + noise1, dim=2)), value)
        return attn

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 1, 3)
k1 = torch.randn(2, 1, 3)
v1 = torch.randn(2, 1, 3)
bias1 = torch.randn(1, 1)
noise1 = torch.randn(1, 2)
