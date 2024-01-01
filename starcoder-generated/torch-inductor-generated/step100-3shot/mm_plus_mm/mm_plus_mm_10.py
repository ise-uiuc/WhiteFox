
class Model(torch.nn.Module):
    def forward(self, input):
        t11 = torch.mm(input, input)
        t12 = torch.mm(input, input)
        t13 = torch.mm(input, input)
        t14 = torch.mm(input, input)
        t21 = torch.mm(input, input)
        t22 = torch.mm(input, input)
        t23 = torch.mm(input, input)
        t31 = torch.mm(input, input)
        t32 = torch.mm(input, input)
        t33 = torch.mm(input, input)
        t34 = torch.mm(input, input)
        t35 = torch.mm(input, input)
        t41 = torch.mm(input, input)
        t42 = torch.mm(input, input)
        t43 = torch.mm(input, input)
        t44 = torch.mm(input, input)
        t45 = torch.mm(input, input)
        t51 = torch.mm(input, input)
        t52 = torch.mm(input, input)
        t53 = torch.mm(input, input)
        out1 = t11 + t12 + t13 + t14 + t21 + t22 + t23
        out2 = t31 + t32 + t33 + t34 + t35
        out3 = t41 + t42 + t43 + t44 + t45
        out4 = t51 + t52 + t53
        return out1 + out2 + out3 + out4
# Inputs to the model
input = torch.randn(100, 100)
