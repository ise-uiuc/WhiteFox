
class Model(torch.nn.Module):
    def forward(self, tensor, tensor2):
        t0 = torch.mm(tensor, tensor)
        t1 = torch.mm(tensor, tensor2)
        t3 = t0 + t1
        return torch.mm(t0, t3)
# Inputs to the model
tensor = torch.randn(20, 20)
tensor2 = torch.randn(20, 20)
