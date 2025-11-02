@echo off
python snp_analysis.py --model auto_enc_non_linear_masked
python snp_analysis.py --model auto_enc_non_linear_masked --center
python snp_analysis.py --model auto_enc_non_linear_masked --center --standardize



