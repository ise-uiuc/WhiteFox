
class Model(torch.nn.Module):
    def forward(self, input_one, input_two, input_three, input_four):
        torch.mul(torch.mm(input_one, input_two),torch.mm(input_three, input_four))
# Inputs to the model
input_one = torch.randn(16, 16)
input_two = torch.randn(16, 16)
input_three = torch.randn(16, 16)
input_four = torch.randn(16, 16)
