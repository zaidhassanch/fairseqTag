import torch
import torch.nn as nn

import math

print("myFC2 imported")

class myFC2(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(myFC2, self).__init__()

		self.groups = in_channels / (math.sqrt(in_channels) if int(math.sqrt(in_channels)) == math.sqrt(in_channels) else math.sqrt(in_channels/2))

		if int(self.groups) != self.groups:
			print("number of groups in my_fc2 not appropriate")

		print(in_channels)
		print(out_channels)
		print(self.groups)

		self.groups = int(self.groups)
		print(self.groups)

		self.reps = 4
		self.copies = out_channels / in_channels

		if int(self.copies) != self.copies:
			print("number of copies in my_fc2 not appropriate")

		self.copies = int(self.copies)

		self.conv1   = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels*self.reps,kernel_size=1,groups=self.groups)
		self.softmax = torch.nn.Softmax(dim=-1)

		self.in_channels = in_channels
		self.out_channels = out_channels


	def forward(self, input_in):

		# old computation
		# x = linear1(input_in)

		# print("initial shape myFC2", input_in.shape)

		input_in = input_in.transpose(0, 1)

		# new computation
		x = torch.transpose(input_in, 1, 2).unsqueeze(3)

		x = self.conv1(x)

		s = x.squeeze()[:, 0::4].transpose(1, 2)

		x = x.transpose(1, 2)

		x = x.reshape(x.shape[0], x.shape[1], self.reps, self.out_channels).transpose(2, 3)

		x = x.reshape(-1, self.out_channels, self.reps)

		x_soft = self.softmax(x.transpose(1, 2)).transpose(1, 2)


		# old
		# attention_scores = torch.bmm(x, x.transpose(1, 2).contiguous())

		attention_KTV = torch.bmm(x_soft.transpose(1,2).contiguous(), x)

		# old
		# attention_scores = attention_scores.view(x.shape[0] * x.shape[1],  -1)
		# attention_weights = self.softmax(attention_scores)
		# attention_weights = attention_weights.view(x.shape[0], x.shape[1], x.shape[1])

		mix = torch.bmm(x, attention_KTV)

		mix = mix.reshape(s.shape[0], s.shape[1], s.shape[2], -1)

		out_attn_val = mix.sum(3)


		# print(x.shape, s.shape, attention_scores.shape, attention_weights.shape, mix.shape, out_attn_val.shape)

		# exit()

		x_out = s + out_attn_val

		x_out = x_out.transpose(0, 1)

		# print("starting to run my_fc2 and now exiting")
		# print(x.shape, s.shape, x_soft.shape, attention_KTV.shape, mix.shape, out_attn_val.shape, x_out.shape)
		# exit()

		return x_out
