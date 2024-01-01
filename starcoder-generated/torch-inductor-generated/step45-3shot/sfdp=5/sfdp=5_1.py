
import torch
import torch.nn as nn
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 80
        self.dim = 40
    def forward(self, input):
        conv1 = nn.Conv1d(input.shape[1], 50, 32)
        conv1 = nn.ReLU(conv1(input))
        conv1 = nn.Conv1d(50, 50, 32, stride=2)
        conv1 = nn.ReLU(conv1(conv1))
        conv1 = nn.Conv1d(50, 50, 32)
        conv1 = nn.ReLU(conv1(conv1))
        conv1 = nn.Conv1d(input.shape[1], 50, 32)
        conv2 = nn.ReLU(conv1(input))
        conv2 = nn.Conv1d(50, 50, 32, stride=2)
        conv2 = nn.ReLU(conv2(conv2))
        conv2 = nn.Conv1d(50, 50, 32)
        conv2 = nn.ReLU(conv2(conv2))
        conv2 = nn.Conv1d(50, 50, 32)
        conv2 = nn.ReLU(conv2(conv2))
        conv2 = nn.Conv1d(input.shape[1], 50, 32)
        conv3 = nn.ReLU(conv2(input))
        conv3 = nn.Conv1d(50, 50, 32, stride=2)
        conv3 = nn.ReLU(conv3(conv3))
        conv3 = nn.Conv1d(50, 50, 32)
        conv3 = nn.ReLU(conv3(conv3))
        conv3 = nn.Conv1d(50, 50, 32)
        conv3 = nn.ReLU(conv3(conv3))
        conv3 = nn.Conv1d(50, 50, 32)
        concat = torch.cat((conv1, conv2, conv3))
        return concat
# Inputs to the model
input = torch.randn(80, 1, 40)
