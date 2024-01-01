
class Model(torch.nn.Module):
    def forward(self, input):
        t11 = torch.mm(input[0:, 0:], input[0:, 0:])
        t12 = torch.mm(input[0:, 0:], input[0:, 0:])
        t13 = torch.mm(input[0:, 0:], input[0:, 0:])
        t21 = torch.mv(input[0:, 0:], input[0:, 0:])
        t22 = torch.mv(input[0:, 0:], input[0:, 0:])
        t23 = torch.mv(input[0:, 0:], input[0:, 0:])
        t31 = torch.mv(input[0:, 0:], input[0:, 0:])
        t32 = torch.mv(input[0:, 0:], input[0:, 0:])
        t33 = torch.mv(input[0:, 0:], input[0:, 0:])
        return t11 + t12 + t13 + t21 + t22 + t23 + t31 + t32 + t33
# Inputs to the model
input = torch.randn(3, 3)
