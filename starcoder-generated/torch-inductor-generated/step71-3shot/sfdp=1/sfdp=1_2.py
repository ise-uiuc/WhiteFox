
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input):
        q = torch.randn(input.shape[0], input.shape[1], input.shape[1])
        k = torch.randn(input.shape[0], input.shape[1], input.shape[1])
        inv_scale = 1e-5
        dropout_p = 0.1
        v = torch.randn(input.shape[0], input.shape[1], input.shape[1])
        return torch.matmul(q, k) + torch.matmul(v, k)

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(4, 20, 128)
