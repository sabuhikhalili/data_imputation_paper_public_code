@echo off
python synthetic_data_analysis.py --model auto_enc_masked_reg 
python synthetic_data_analysis.py --model auto_enc_masked_reg --center
python synthetic_data_analysis.py --model auto_enc_masked_reg --center --standardize


