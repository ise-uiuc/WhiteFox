
class Model(torch.nn.Module):
    def forward(self, in_tensor, in_tensor_1):
        t1 = torch.reshape(in_tensor, (32, 16, 3, 3))
        t2 = torch.reshape(in_tensor_1, (32, 16, 3, 3))
        t3 = torch.cat([t1, torch.reshape(t2, (32, 15, 3, 3))], 1)
        return t3
# Inputs to the model
in_tensor = torch.randn(3, 3)
in_tensor_1 = torch.randn(3, 3)
