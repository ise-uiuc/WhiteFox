
class Model(torch.nn.Module):
    def forward(self, model_input, model_input_2, model_input_3, model_input_4):
        v1 = torch.mm(model_input, model_input)
        v2 = torch.mm(model_input_3, model_input_4)
        return v1 * v2
# Inputs to the model
model_input = torch.randn(10, 10)
model_input_2 = torch.randn(33, 33)
model_input_3 = torch.randn(19, 19)
model_input_4 = torch.randn(87, 87)
