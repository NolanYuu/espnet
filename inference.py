#!/usr/bin/env python3
import os
import re
import time
import numpy as np
import torch
import soundfile as sf
import time
from pathlib import Path
from tqdm import tqdm
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from espnet2.bin.tts_inference import Text2Speech


if __name__ == "__main__":
    print()
    if not os.path.exists("./inference/"):
        os.mkdir("./inference")

    vocoder_tag = "ljspeech_multi_band_melgan.v2" #@param ["ljspeech_parallel_wavegan.v1", "ljspeech_full_band_melgan.v2", "ljspeech_multi_band_melgan.v2"] {type:"string"}
    vocoder = load_model(download_pretrained_model(vocoder_tag)).to("cuda").eval()
    vocoder.remove_weight_norm()

    models = [
        # "transformer", 
        # "tacotron2", 
        # "fastspeech",
        "fastspeech2",
        # "conformer_fastspeech2",
    ]
    for model in models:
        text2speech = Text2Speech("./egs2/ljspeech/tts1/exp/tts_train_{}_raw_phn_tacotron_g2p_en_no_space/config.yaml".format(model), "./egs2/ljspeech/tts1/exp/tts_train_{}_raw_phn_tacotron_g2p_en_no_space/valid.loss.best.pth".format(model), device="cuda")
        fs = text2speech.fs
        text2speech.spc2wav = None
        texts = [
            # "they state that they are compelled by an imperative sense of duty to advert in terms of decided condemnation to the lamentable condition of the prisons of the city of London,",
            # "they state that they are compelled by an imperative sense of duty to advert in terms of decided condemnation to the lamentable condition of the prisons of the city of London, The prison officials appear to be on the side of the inspectors, to the great dissatisfaction of the corporation, who claimed the full allegiance and support of its servants. "
            # "The Court in addition to the proper use of its judicial functions has improperly set itself up as a third house of the Congress. If, for instance, any one of the six justices of the Supreme Court now over the age of seventy should retire as provided under the plan, "
            # "If such a plan is good for the lower courts it certainly ought to be equally good for the highest court from which there is no appeal. Is it a dangerous precedent for the Congress to change the number of the justices? The Congress has always had, and will have, that power.",
            # "The prison officials appear to be on the side of the inspectors, to the great dissatisfaction of the corporation, who claimed the full allegiance and support of its servants.",
            # "in some yards",
            # "We have, therefore,",
            "The feature matching loss is a learned similarity metric measured by the difference in features of the discriminator between a ground truth sample and a generated sample",
        ]
        with torch.no_grad():
            for i, text in tqdm(enumerate(texts)):
                text = re.sub(r"([,.?!])(?!\s)", r"\1 ", text).rstrip()
                # print(time.time())
                _, c, *_ = text2speech(text)
                # sf.write("/nolan/inference/test_gl.wav", wav.data.cpu().numpy(), text2speech.fs, "PCM_16")
                np.save("./inference/{}_{:02d}.mel.npy".format(model, i), c.data.cpu().numpy())
                wav = vocoder.inference(c)
                sf.write("./inference/{}_{:02d}.wav".format(model, i), wav.data.cpu().numpy(), fs, "PCM_16")

    # # fastspeech2 = Text2Speech("/nolan/test/espnet/egs2/ljspeech/tts1/tmp_save/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/config.yaml", "/nolan/test/espnet/egs2/ljspeech/tts1/tmp_save/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/valid.loss.best.pth", device="cuda")
    # gradtts = Text2Speech("/nolan/test/espnet/egs2/ljspeech/tts1/tmp_save/tts_train_gradtts_raw_phn_tacotron_g2p_en_no_space/config.yaml", "/nolan/test/espnet/egs2/ljspeech/tts1/tmp_save/tts_train_gradtts_raw_phn_tacotron_g2p_en_no_space/train.loss.best.pth", device="cuda")
    # # gradtts = Text2Speech("/nolan/exp/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/config.yaml", "/nolan/exp/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/test.pth", device="cuda")
    # fs = gradtts.fs
    # # fastspeech2.spc2wav = None
    # gradtts.spc2wav = None
    # # gradfastspeech2 = Text2Speech("/nolan/test/espnet/egs2/ljspeech/tts1/exp/tts_train_gradfastspeech2_raw_phn_tacotron_g2p_en_no_space/config.yaml", "/nolan/test/espnet/egs2/ljspeech/tts1/exp/tts_train_gradfastspeech2_raw_phn_tacotron_g2p_en_no_space/valid.loss.best.pth", device="cuda")
    # # fs = gradfastspeech2.fs
    # # gradfastspeech2.spc2wav = None
    # texts = [
    #         "The encoding of each struct field can be customized by the format string stored under the key in the struct field's tag. The format string gives the name of the field, possibly followed by a comma separated list of options. "
    #         # "they state that they are compelled by an imperative sense of duty to advert in terms of decided condemnation to the lamentable condition of the prisons of the city of London,",
    #         # "they state that they are compelled by an imperative sense of duty to advert in terms of decided condemnation to the lamentable condition of the prisons of the city of London, The prison officials appear to be on the side of the inspectors, to the great dissatisfaction of the corporation, who claimed the full allegiance and support of its servants. The Court in addition to the proper use of its judicial functions has improperly set itself up as a third house of the Congress. If, for instance, any one of the six justices of the Supreme Court now over the age of seventy should retire as provided under the plan, Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples."
    #         # "The Court in addition to the proper use of its judicial functions has improperly set itself up as a third house of the Congress. If, for instance, any one of the six justices of the Supreme Court now over the age of seventy should retire as provided under the plan, "
    #         # "If such a plan is good for the lower courts it certainly ought to be equally good for the highest court from which there is no appeal. Is it a dangerous precedent for the Congress to change the number of the justices? The Congress has always had, and will have, that power.",
    #         # "The prison officials appear to be on the side of the inspectors, to the great dissatisfaction of the corporation, who claimed the full allegiance and support of its servants.",
    #         # "in some yards",
    #         # "We have, therefore,",
    #         # "The feature matching loss is a learned similarity metric measured by the difference in features of the discriminator between a ground truth sample and a generated sample",
    #     ]
    # with torch.no_grad():
    #     for i, text in tqdm(enumerate(texts)):
    #         text = re.sub(r"([,.?!])(?!\s)", r"\1 ", text).rstrip()
    #         _, c, *_ = gradtts(text)
    #         np.save("/nolan/inference/gradtts_{:02d}.mel.npy".format(i), c.data.cpu().numpy())
    #         c = gradtts.tts.decode_inference(c)
    #         np.save("/nolan/inference/gradtts_{:02d}.mel.npy".format(i+1), c.data.cpu().numpy())
            
    #         wav = vocoder.inference(c)
    #         sf.write("/nolan/inference/test.wav", wav.data.cpu().numpy(), fs, "PCM_16")