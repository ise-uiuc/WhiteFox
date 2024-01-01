
class Model(torch.nn.Module):
    def forward(self, input_tensor):
        t1 = torch.cat(input_tensors, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
__input_tensors__ = [torch.randn(1, 3, 64, 64) for _ in range(5)]
