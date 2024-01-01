
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        k1 = torch.reshape(x2, (1, 192, 47))
        k2 = torch.zeros([1, 12, 15, 5], dtype=torch.float32, layout=torch.strided, device=torch.device("cpu"))
        v1 = torch.bmm(k1, k2)
        v2 = torch.sub(x1, v1)
        k3 = torch.reshape(x1, (1, 22, 144))
        k4 = torch.tensor([[-0.6940]], dtype=torch.float32, layout=torch.strided, device=torch.device("cpu"))

        return torch.nn.functional.relu(v2)
# Inputs to the model
x1 = torch.randn(1, 22, 145)
x2 = torch.randn(1, 192, 47)
