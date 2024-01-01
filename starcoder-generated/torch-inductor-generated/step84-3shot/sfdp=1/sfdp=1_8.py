
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q = x1
        k = torch.transpose(x2, 1, 2)
        x = torch.matmul(q, k)
        scale = 1 / np.sqrt(q.shape[-1])
        x = x * scale
        x = x.softmax(dim=-1)
        scale_x = torch.nn.functional.dropout(x, p=0.2)
        d = torch.matmul(scale_x, x2)
        return d

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 64, 512, 64)
x2 = torch.randn(1, 512, 64, 64)
