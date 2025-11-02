@echo off
python synthetic_data_analysis.py --model auto_enc
python synthetic_data_analysis.py --model auto_enc --center
python synthetic_data_analysis.py --model auto_enc --center --standardize

python synthetic_data_analysis.py --model auto_enc_masked 
python synthetic_data_analysis.py --model auto_enc_masked --center
python synthetic_data_analysis.py --model auto_enc_masked --center --standardize

python synthetic_data_analysis.py --model auto_enc_masked_reg 
python synthetic_data_analysis.py --model auto_enc_masked_reg --center
python synthetic_data_analysis.py --model auto_enc_masked_reg --center --standardize


