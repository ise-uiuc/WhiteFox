
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([3.0]))
 
    def forward(self, q1, k1, v1):
        v = q1.matmul(k1.transpose(-2, -1))
        scale_v = self.scale_factor * v
        softmax = scale_v.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax, p=0.5)
        out = dropout.matmul(v1)
        return out

# Initializing the model
m = Model()

