
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input):
        c = input[0].size(2)
        x = [input[i][:, :, :c-2*i] for i in range(16)]
        x = torch.cat(x, dim=2)
        return torch.cat([input[0], x], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
input = [torch.randn(16, 15, 23+i) for i in range(16)]
