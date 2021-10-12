# ESPnet

ESPnet is an open-source speech processing toolkit

## Installation

Please refer to the [tutorial page](https://espnet.github.io/espnet/installation.html) 

For easier access, there is a shared docker image **espnet-nolan** in our [AiStation](https://10.100.112.79:32206/index.html), just run it, follow the commands below, and all the components will be available.

```
# in espnet-nolan
cd ./tools
make
```



## Train

For Tacotron2 training, just follow the command below.

```
cd ./egs2/ljspeeech/tts1
./run.sh
```

*Notice that if you want to train FastSpeech2, the training of a teacher model (Tacotron2 or Transformer-TTS) is necessary.*

For FastSpeech2 training, follow the command below.

```
./run.sh --stage 7 \
         --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
         --inference_args "--use_teacher_forcing true" \
         --test_sets "tr_no_dev dev eval1"
         
./run.sh --stage 5 \
         --train_config conf/tuning/train_fastspeech2.yaml \
         --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave \
         --tts_stats_dir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave/stats \
         --write_collected_feats true
```



## Inference

To test the model with an arbitrary given text, just run espnet/inference.py, and modify the texts.

```
cd ../../..
python inference.py
```

