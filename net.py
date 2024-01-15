import torch
import torch.nn.functional as F


class ChessDQN(torch.nn.Module):
  def __init__(self):
    super(ChessDQN, self).__init__()
    # First convolutional layer
    self.conv_current = torch.nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv_target = torch.nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1)
    # Second convolutional layer
    self.conv2 = torch.nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.fc = torch.nn.Linear(64 * 8 * 8, 64 * 64)

  def forward(self, current, target):
    current = F.relu(self.conv_current(current))
    target = F.relu(self.conv_target(target))
    x = torch.concat((current, target), dim=1)
    x = F.relu(self.conv2(x))
    x = torch.flatten(x, start_dim=1)
    x = self.fc(x)
    return x.view(-1, 64, 64)
