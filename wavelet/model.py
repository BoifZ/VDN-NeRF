# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch.nn as nn

from networks.encoders import ResnetEncoder, DenseEncoder, MobileNetV2Encoder
from networks.decoders import DecoderWave, DecoderWave224, Decoder, Decoder224, SparseDecoderWave

class InfModel(nn.Module):
    def __init__(self, opts):
        super(Model, self).__init__()
        print("Building model ", end="")
        # self.encoder = 
        decoder_width = 0.5
        if opts.encoder_type == "densenet":
            self.encoder = DenseEncoder(normalize_input=opts.normalize_input, pretrained=opts.pretrained_encoder)
        elif opts.encoder_type == "resnet":
            self.encoder = ResnetEncoder(num_layers=opts.num_layers, pretrained=opts.pretrained_encoder,
                                         normalize_input=opts.normalize_input)
        elif opts.encoder_type == "mobilenet":
            self.encoder = MobileNetV2Encoder(pretrained=opts.pretrained_encoder, use_last_layer=True,
                                              normalize_input=opts.normalize_input)
        elif opts.encoder_type == "mobilenet_light":
            self.encoder = MobileNetV2Encoder(pretrained=opts.pretrained_encoder, use_last_layer=False,
                                              normalize_input=opts.normalize_input)
        else:
            raise NotImplementedError


    def DenseEncoder(self, normalize_input=True, num_layers=161, pretrained=False):
        import torchvision.models as models

        model_dict = {161: models.densenet161,
                      121: models.densenet121,
                      201: models.densenet201,
                      169: models.densenet169}

        assert num_layers in model_dict, "Can't use any number of layers, should use from 121, 161, 169, 201"

        self.original_model = models.densenet161( pretrained=pretrained )
        self.normalize_input = normalize_input

        import numpy as np
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.num_ch_enc = [2208, 384, 192, 96, 96]
        self.num_ch_enc.reverse()

    def forward(self, x, threshold=-1):
        if self.normalize_input:
            for t, m, s in zip(x, self.mean, self.std):
                t.sub(m).div(s)

        features = [x]
        cnt = 1
        for k, v in self.original_model.features._modules.items(): 
            if cnt > 4: continue
            cnt += 1
            features.append( v(features[-1]) )
        # return features[3], features[4], features[6], features[8], features[11]
        return features[3], features[4]
        
        x = self.encoder(x)
        for y in x:
            print(y.shape)
        return x


class Model(nn.Module):
    def __init__(self, opts):
        super(Model, self).__init__()

        print("Building model ", end="")

        decoder_width = 0.5
        if opts.encoder_type == "densenet":
            self.encoder = DenseEncoder(normalize_input=opts.normalize_input, pretrained=opts.pretrained_encoder)
        elif opts.encoder_type == "resnet":
            self.encoder = ResnetEncoder(num_layers=opts.num_layers, pretrained=opts.pretrained_encoder,
                                         normalize_input=opts.normalize_input)
        elif opts.encoder_type == "mobilenet":
            self.encoder = MobileNetV2Encoder(pretrained=opts.pretrained_encoder, use_last_layer=True,
                                              normalize_input=opts.normalize_input)
        elif opts.encoder_type == "mobilenet_light":
            self.encoder = MobileNetV2Encoder(pretrained=opts.pretrained_encoder, use_last_layer=False,
                                              normalize_input=opts.normalize_input)
        else:
            raise NotImplementedError

        print("using {} encoder".format(opts.encoder_type))

        self.use_sparse = False

        if opts.use_wavelets:
            try:
                if opts.use_sparse:
                    self.use_sparse = True
                    if opts.use_224:
                        raise NotImplementedError
            except AttributeError:
                opts.use_sparse = False
                self.use_sparse = False

            if opts.use_sparse:
                self.decoder = SparseDecoderWave(enc_features=self.encoder.num_ch_enc, decoder_width=decoder_width)
            else:
                if opts.use_224:
                    decoder_wave = DecoderWave224
                else:
                    decoder_wave = DecoderWave

                self.decoder = decoder_wave(enc_features=self.encoder.num_ch_enc, decoder_width=decoder_width,
                                            dw_waveconv=opts.dw_waveconv,
                                            dw_upconv=opts.dw_upconv)
        else:
            if opts.use_224:
                decoder = Decoder224
            else:
                decoder = Decoder
            self.decoder = decoder(enc_features=self.encoder.num_ch_enc,
                                   is_depthwise=(opts.dw_waveconv or opts.dw_upconv))

    def forward(self, x, threshold=-1):
        x = self.encoder(x)
        # for y in x:
        #     print(y.shape)
        if self.use_sparse:
            return self.decoder(x, threshold)
        else:
            return self.decoder(x)
