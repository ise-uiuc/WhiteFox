
class Model(torch.nn.Module):
    def forward(self, *input_tensors):
        t1 = torch.cat(input_tensors, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        return t2[:, 0:size]
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
x5 = torch.randn(1, 3, 64, 64)
