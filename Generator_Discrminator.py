import torch

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()

        strides = [1, 2, 2, 2]
        padding = [0, 1, 1, 1]
        channels = [input_size,
                    256, 128, 64, 1]  # 1表示一维
        kernels = [4, 3, 4, 4]

        model = []
        for i, stride in enumerate(strides):
            model.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    kernel_size=kernels[i],
                    padding=padding[i]
                )
            )

            if i != len(strides) - 1:
                model.append(
                    nn.BatchNorm2d(channels[i + 1])
                )
                model.append(
                    nn.ReLU()
                )
            else:
                model.append(
                    nn.Tanh()
                )

        self.main = nn.Sequential(*model)

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        strides = [2, 2, 2]
        padding = [1, 1, 1]
        channels = [input_size,
                    64, 128, 256]  # 1表示一维
        kernels = [4, 4, 4]

        model = []
        for i, stride in enumerate(strides):
            model.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    kernel_size=kernels[i],
                    padding=padding[i]
                )
            )
            model.append(
                nn.BatchNorm2d(channels[i + 1])
            )
            model.append(
                nn.LeakyReLU(0.2)
            )

        self.main = nn.Sequential(*model)
        self.D = nn.Sequential(
            nn.Linear(3 * 3 * 256, 1),
            nn.Sigmoid()
        )
        self.C = nn.Sequential(
            nn.Linear(3 * 3 * 256, 10),
            nn.Softmax(dim=1)
        )

    #         self.L = nn.Sequential(
    #             nn.Linear(3 * 3 * 256, 2),
    #         )

    def forward(self, x):
        x = self.main(x).view(x.shape[0], -1)
        return self.D(x), self.C(x)  # self.L(x)