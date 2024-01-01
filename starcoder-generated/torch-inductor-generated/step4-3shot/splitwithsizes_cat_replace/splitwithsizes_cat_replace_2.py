
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def split(self, x1, split_sizes, dim):
        return torch.split(x1, split_sizes, dim)
    
    def forward(self, x1):
        split_sizes = [1, 2, 3, 8]
        split_tensors = self.split(x1, split_sizes, dim)
        v_out = []
        for i in range(len(split_sizes)):
            v_out.append(split_tensors[i])
        v6 = torch.cat(v_out, dim)
        return v_out[2]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 8)
