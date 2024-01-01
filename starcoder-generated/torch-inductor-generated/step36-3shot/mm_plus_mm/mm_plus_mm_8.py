
class Model(torch.nn.Module):
    def forward(self, left, right):
        mm1 = torch.mm(left, right)
        left = torch.mm(left, right)
        right = torch.mm(left, right)
        mm2 = left + right
        mm1 = torch.mm(left, right)
        return mm1 + mm2
# Inputs to the model
left = torch.randn(20, 10)
right = torch.randn(10, 20)
