
class Model(torch.nn.Module):
    def forward(self, tensor):
        t1 = torch.mm(tensor[:32, :32], tensor[32:, :32])
        t2 = torch.mm(tensor[32:, :32], tensor[:32, 32:])
        t3 = t1 + t2
        return t3
# Inputs to the model
tensor = torch.randn(64, 64)
