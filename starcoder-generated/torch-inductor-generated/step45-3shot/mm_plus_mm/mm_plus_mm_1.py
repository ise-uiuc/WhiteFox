
import torch
# The problem we are trying to solve is computing a tensor
# of shape `(batch_size, num_steps, hidden_size)` where
# for each instance we have
# [input_steps, batch_size, hidden_size] input and
# [batch_size, hidden_size, hidden_size] weights all
# multiplied together.

class Model(torch.nn.Module):
    def forward(self, x, w):
        shape = w.shape
        x = x.reshape((*shape, 1))
        y = x.squeeze() * w
        return y
# Inputs to the model
x = torch.randn(10, 3, 4)
w = torch.randn(3, 4, 5)
