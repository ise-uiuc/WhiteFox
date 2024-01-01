
class Model(torch.nn.Module):
    def forward(self, input):
        output = 0

        for i in input:
            for j in i:
                if (j > 0):
                    output = output + 1

        return output
# Inputs to the model
input = torch.randn((3,5))
