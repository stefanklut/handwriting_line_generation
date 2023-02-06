#!/bin/bash

python train.py -c configs/cf_PageXML_hwr_cnnOnly_batchnorm_aug.json
if [[ $? -ne 0 ]]; then
    echo "cf_PageXML_hwr_cnnOnly_batchnorm_aug.json has errored, stopping program"
    exit 1
fi
python train.py -c configs/cf_PageXML_auto_2tight_newCTC.json
if [[ $? -ne 0 ]]; then
    echo "cf_PageXML_auto_2tight_newCTC.json has errored, stopping program"
    exit 1
fi
python train.py -c configs/cf_PageXMLslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG.json
if [[ $? -ne 0 ]]; then
    echo "cf_PageXMLslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG.json has errored, stopping program"
    exit 1
fi