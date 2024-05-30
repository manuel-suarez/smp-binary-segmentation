import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = EncoderBlock(in_channels=3, out_channels=64)
        self.block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.block4 = EncoderBlock(in_channels=256, out_channels=512)
        self.block5 = EncoderBlock(in_channels=512, out_channels=1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        r1 = self.block1(x)
        r2 = self.block2(self.maxpool(r1))
        r3 = self.block3(self.maxpool(r2))
        r4 = self.block4(self.maxpool(r3))
        x = self.block5(self.maxpool(r4))

        return x, [r1,r2,r3,r4]

class DecoderUpBlock(nn.Module):
    def __init__(self, upsample_size, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.upsampling = nn.Upsample(size=(upsample_size, upsample_size))
        self.upconv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2)
        # concatenation
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()

    def forward(self, x, r):
        x = self.upsampling(x)
        x = self.upconv1(x)
        x = torch.cat([x, r], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size=28):
        super().__init__()
        self.block4 = DecoderUpBlock(upsample_size=input_size+1, in_channels=1024, out_channels=512)
        input_size = input_size * 2
        self.block3 = DecoderUpBlock(upsample_size=input_size+1, in_channels=512, out_channels=256)
        input_size = input_size * 2
        self.block2 = DecoderUpBlock(upsample_size=input_size+1, in_channels=256, out_channels=128)
        input_size = input_size * 2
        self.block1 = DecoderUpBlock(upsample_size=input_size+1, in_channels=128, out_channels=64)
        self.head = SegmentationHead()

    def forward(self, x, r):
        x = self.block4(x, r[3])
        x = self.block3(x, r[2])
        x = self.block2(x, r[1])
        x = self.block1(x, r[0])
        x = self.head(x)

        return x

class UNet(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        self.encoder = Encoder()
        # 4 layers
        input_size = input_size // 2
        input_size = input_size // 2
        input_size = input_size // 2
        self.decoder = Decoder(input_size)

    def forward(self, x):
        x, r = self.encoder(x)
        x = self.decoder(x, r)

        return x

import unittest
class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.input_channels = 3
        self.input_size = 224
        self.block_channels = [64, 128, 256, 512, 1024]
        self.block_sizes = [224, 112, 56, 28, 14]

    def test_encoder_block1(self):
        input = torch.randn((1, 3, self.block_sizes[0], self.block_sizes[0]))
        output = Encoder().block1(input)
        self.assertEqual(output.shape, (1, self.block_channels[0], self.block_sizes[0], self.block_sizes[0]))

    def test_encoder_block2(self):
        input = torch.randn((1, self.block_channels[0], self.block_sizes[1], self.block_sizes[1]))
        output = Encoder().block2(input)
        self.assertEqual(output.shape, (1, self.block_channels[1], self.block_sizes[1], self.block_sizes[1]))

    def test_encoder_block3(self):
        input = torch.randn((1, self.block_channels[1], self.block_sizes[2], self.block_sizes[2]))
        output = Encoder().block3(input)
        self.assertEqual(output.shape, (1, self.block_channels[2], self.block_sizes[2], self.block_sizes[2]))

    def test_encoder_block4(self):
        input = torch.randn((1, self.block_channels[2], self.block_sizes[3], self.block_sizes[3]))
        output = Encoder().block4(input)
        self.assertEqual(output.shape, (1, self.block_channels[3], self.block_sizes[3], self.block_sizes[3]))

    def test_encoder_block5(self):
        input = torch.randn((1, self.block_channels[3], self.block_sizes[4], self.block_sizes[4]))
        output = Encoder().block5(input)
        self.assertEqual(output.shape, (1, self.block_channels[4], self.block_sizes[4], self.block_sizes[4]))

    def test_encoder_blocks(self):
        input = torch.randn((1, self.input_channels, self.input_size, self.input_size))
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        output1 = Encoder().block1(input)
        self.assertEqual(output1.shape, (1, self.block_channels[0], self.block_sizes[0], self.block_sizes[0]))
        output2 = Encoder().block2(maxpool(output1))
        self.assertEqual(output2.shape, (1, self.block_channels[1], self.block_sizes[1], self.block_sizes[1]))
        output3 = Encoder().block3(maxpool(output2))
        self.assertEqual(output3.shape, (1, self.block_channels[2], self.block_sizes[2], self.block_sizes[2]))
        output4 = Encoder().block4(maxpool(output3))
        self.assertEqual(output4.shape, (1, self.block_channels[3], self.block_sizes[3], self.block_sizes[3]))
        output5 = Encoder().block5(maxpool(output4))
        self.assertEqual(output5.shape, (1, self.block_channels[4], self.block_sizes[4], self.block_sizes[4]))

    def test_encoder_module(self):
        input = torch.randn((1, 3, 224, 224))
        output, _ = Encoder()(input)
        self.assertEqual(output.shape, (1, 1024, 14, 14))

class TestDecoder(unittest.TestCase):
    def test_decoder_upblock4(self):
        input = torch.randn((1, 1024, 14, 14))
        skip = torch.rand((1, 512, 28, 28))
        output = Decoder().block4(input, skip)
        self.assertEqual(output.shape, (1, 512, 28, 28))

    def test_decoder_upblock3(self):
        input = torch.randn((1, 512, 28, 28))
        skip = torch.rand((1, 256, 56, 56))
        output = Decoder().block3(input, skip)
        self.assertEqual(output.shape, (1, 256, 56, 56))

    def test_decoder_upblock2(self):
        input = torch.randn((1, 256, 56, 56))
        skip = torch.rand((1, 128, 112, 112))
        output = Decoder().block2(input, skip)
        self.assertEqual(output.shape, (1, 128, 112, 112))

    def test_decoder_upblock1(self):
        input = torch.randn((1, 128, 112, 112))
        skip = torch.rand((1, 64, 224, 224))
        output = Decoder().block1(input, skip)
        self.assertEqual(output.shape, (1, 64, 224, 224))

    def test_segmentation_head(self):
        input = torch.randn((1, 64, 224, 224))
        output = SegmentationHead()(input)
        self.assertEqual(output.shape, (1, 1, 224, 224))

    def test_decoder_blocks(self):
        input = torch.randn((1, 3, 224, 224))
        output, skips = Encoder()(input)

        output = Decoder().block4(output, skips[3])
        self.assertEqual(output.shape, (1, 512, 28, 28))
        output = Decoder().block3(output, skips[2])
        self.assertEqual(output.shape, (1, 256, 56, 56))
        output = Decoder().block2(output, skips[1])
        self.assertEqual(output.shape, (1, 128, 112, 112))
        output = Decoder().block1(output, skips[0])
        self.assertEqual(output.shape, (1, 64, 224, 224))
        output = SegmentationHead()(output)
        self.assertEqual(output.shape, (1, 1, 224, 224))

    def test_decoder_module(self):
        input = torch.randn((1, 3, 224, 224))
        output, r = Encoder()(input)
        output = Decoder()(output, r)
        self.assertEqual(output.shape, (1, 1, 224, 224))

class TestUNet(unittest.TestCase):
    def test_unet(self):
        model = UNet()
        input = torch.randn((1, 3, 224, 224))
        output = model(input)
        self.assertEqual(output.shape, (1, 1, 224, 224))

if __name__ == '__main__':
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    input = torch.randn((1, 3, 224, 224))

    model = Encoder().block1
    output1 = model(input)
    print("Block 1: ", output1.shape)

    model = Encoder().block2
    output2 = model(maxpool(output1))
    print("Block 2: ", output2.shape)

    model = Encoder().block3
    output3 = model(maxpool(output2))
    print("Block 3: ", output3.shape)

    model = Encoder().block4
    output4 = model(maxpool(output3))
    print("Block 4: ", output4.shape)

    model = Encoder().block5
    output5 = model(maxpool(output4))
    print("Block 5: ", output5.shape)

    model = Encoder()
    output, _ = model(input)
    print("Encoder: ", output.shape)

    model = Decoder().block4
    output = model(output, output4)
    print("Up block 4: ", output.shape)

    model = Decoder().block3
    output = model(output, output3)
    print("Up block 3: ", output.shape)

    model = Decoder().block2
    output = model(output, output2)
    print("Up block 2: ", output.shape)

    model = Decoder().block1
    output = model(output, output1)
    print("Up block 1: ", output.shape)

    model = Encoder()
    output, r = model(input)
    model = Decoder()
    output = model(output, r)
    print("Decoder: ", output.shape)

    model = UNet()
    output = model(input)
    print("UNet: ", output.shape)

    unittest.main()
