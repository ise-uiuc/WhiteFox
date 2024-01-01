
class Model(torch.nn.Module):
    def forward(self, input_tensors):
        c = torch.cat(input_tensors, dim=1)
        t = c[:, 0:9223372036854775807]
        s = t[:, 0:self.size]
        x = torch.cat([c, t], dim=1)
        return x

# Initializing the model
m = Model()

# Input tensor
x1 = [torch.randn(1, 10, 1024), torch.randn(1, 10, 1024)]

# Inputs to the model
