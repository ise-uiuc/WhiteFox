
class Model(torch.nn.Module):
    def forward(self, inputs1, inputs2):
        inputs_1 = torch.cat((inputs1, inputs2), 0)
        inputs_2 = torch.cat((inputs2, inputs1), 0)
        out = torch.rand(torch.Size([int(len(inputs1)*2)]))

        temp = torch.mm(inputs_1, inputs_2)
        out = out + temp
        return out
# Inputs to the model
inputs_1 = torch.randn(10, 10)
inputs_2 = torch.randn(10, 10)
