
class Model(torch.nn.Module):
    def forward(self, model_input):
        t1 = torch.mm(model_input, model_input)
        t2 = torch.mm(model_input, model_input)
        t3 = torch.mm(model_input, model_input)
        t4 = torch.mm(model_input, model_input)
        t5 = torch.mm(model_input, model_input)
        t6 = torch.mm(model_input, model_input)
        t7 = torch.mm(model_input, model_input)
        t8 = torch.mm(model_input, model_input)
        t9 = torch.mm(model_input, model_input)
        t10 = torch.mm(model_input, model_input)
        t11 = torch.mm(model_input, model_input)
        t12 = torch.mm(model_input, model_input)
        t13 = torch.mm(model_input, model_input)
        t14 = torch.mm(model_input, model_input)
        t15 = torch.mm(model_input, model_input)
        t16 = torch.mm(model_input, model_input)
        t17 = torch.mm(model_input, model_input)
        t18 = torch.mm(model_input, model_input)
        t19 = torch.mm(model_input, model_input)
        t20 = torch.mm(model_input, model_input)
        return t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15 + t16 + t17 + t18 + t19 + t20
# Inputs to the model
model_input = torch.randn(1000, 1000)
