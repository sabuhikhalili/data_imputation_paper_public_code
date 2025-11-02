@echo off
python synthetic_data_analysis_blocks_2.py --model auto_enc_masked_reg
python synthetic_data_analysis_blocks_2.py --model auto_enc_masked_reg --center
python synthetic_data_analysis_blocks_2.py --model auto_enc_masked_reg --center --standardize

python synthetic_data_analysis_blocks_2.py --model auto_enc_masked
python synthetic_data_analysis_blocks_2.py --model auto_enc_masked --center
python synthetic_data_analysis_blocks_2.py --model auto_enc_masked --center --standardize

python synthetic_data_analysis_blocks_3.py --model auto_enc_masked_reg
python synthetic_data_analysis_blocks_3.py --model auto_enc_masked_reg --center
python synthetic_data_analysis_blocks_3.py --model auto_enc_masked_reg --center --standardize

python synthetic_data_analysis_blocks_3.py --model auto_enc_masked
python synthetic_data_analysis_blocks_3.py --model auto_enc_masked --center
python synthetic_data_analysis_blocks_3.py --model auto_enc_masked --center --standardize

python synthetic_data_analysis_blocks_4.py --model auto_enc_masked_reg
python synthetic_data_analysis_blocks_4.py --model auto_enc_masked_reg --center
python synthetic_data_analysis_blocks_4.py --model auto_enc_masked_reg --center --standardize

python synthetic_data_analysis_blocks_4.py --model auto_enc_masked
python synthetic_data_analysis_blocks_4.py --model auto_enc_masked --center
python synthetic_data_analysis_blocks_4.py --model auto_enc_masked --center --standardize

python synthetic_data_analysis_blocks_2.py --model complete
python synthetic_data_analysis_blocks_2.py --model complete --center
python synthetic_data_analysis_blocks_2.py --model complete --center --standardize

python synthetic_data_analysis_blocks_2.py --model tp_apc
python synthetic_data_analysis_blocks_2.py --model tp_apc --center
python synthetic_data_analysis_blocks_2.py --model tp_apc --center --standardize

python synthetic_data_analysis_blocks_2.py --model tw_apc
python synthetic_data_analysis_blocks_2.py --model tw_apc --center
python synthetic_data_analysis_blocks_2.py --model tw_apc --center --standardize

python synthetic_data_analysis_blocks_3.py --model complete
python synthetic_data_analysis_blocks_3.py --model complete --center
python synthetic_data_analysis_blocks_3.py --model complete --center --standardize

python synthetic_data_analysis_blocks_3.py --model tp_apc
python synthetic_data_analysis_blocks_3.py --model tp_apc --center
python synthetic_data_analysis_blocks_3.py --model tp_apc --center --standardize

python synthetic_data_analysis_blocks_3.py --model tw_apc
python synthetic_data_analysis_blocks_3.py --model tw_apc --center
python synthetic_data_analysis_blocks_3.py --model tw_apc --center --standardize

python synthetic_data_analysis_blocks_4.py --model complete
python synthetic_data_analysis_blocks_4.py --model complete --center
python synthetic_data_analysis_blocks_4.py --model complete --center --standardize

python synthetic_data_analysis_blocks_4.py --model tp_apc
python synthetic_data_analysis_blocks_4.py --model tp_apc --center
python synthetic_data_analysis_blocks_4.py --model tp_apc --center --standardize

python synthetic_data_analysis_blocks_4.py --model tw_apc
python synthetic_data_analysis_blocks_4.py --model tw_apc --center
python synthetic_data_analysis_blocks_4.py --model tw_apc --center --standardize




