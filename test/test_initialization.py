# coding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import importlib
import argparse

import pytest
import numpy


args = argparse.Namespace(
    elayers = 4,
    subsample = "1_2_2_1_1",
    etype = "vggblstmp",
    eunits = 100,
    eprojs = 100,
    dlayers=1,
    dunits=300,
    atype="location",
    aconv_chans=10,
    aconv_filts=100,
    mtlalpha=0.5,
    adim=320,
    dropout_rate=0.0,
    beam_size=3,
    penalty=0.5,
    maxlenratio=1.0,
    minlenratio=0.0,
    verbose = True,
    char_list = [u"あ", u"い", u"う", u"え", u"お"],
    outdir = None
)



def test_lecun_init_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("pytorch is not installed")

    import e2e_asr_attctc_th as m
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    b = model.predictor.ctc.ctc_lo.bias.data.numpy()
    assert numpy.all(b == 0.0)
    w = model.predictor.ctc.ctc_lo.weight.data.numpy()
    numpy.testing.assert_allclose(w.mean(), 0.0, 1e-2, 1e-2)
    numpy.testing.assert_allclose(w.var(), 1.0 / w.shape[1], 1e-2, 1e-2)


def test_lecun_init_chainer():
    import e2e_asr_attctc as m
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    b = model.predictor.ctc.ctc_lo.b.data
    assert numpy.all(b == 0.0)
    w = model.predictor.ctc.ctc_lo.W.data
    numpy.testing.assert_allclose(w.mean(), 0.0, 1e-2, 1e-2)
    numpy.testing.assert_allclose(w.var(), 1.0 / w.shape[1], 1e-2, 1e-2)
    
