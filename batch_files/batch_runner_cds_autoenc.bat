@echo off

python cds_analysis.py --model auto_enc
python cds_analysis.py --model auto_enc_masked 
python cds_analysis.py --model auto_enc_masked_reg

python cds_analysis.py --model auto_enc --center
python cds_analysis.py --model auto_enc_masked --center
python cds_analysis.py --model auto_enc_masked_reg --center

python cds_analysis.py --model auto_enc --center --standardize
python cds_analysis.py --model auto_enc_masked --center --standardize
python cds_analysis.py --model auto_enc_masked_reg --center --standardize

