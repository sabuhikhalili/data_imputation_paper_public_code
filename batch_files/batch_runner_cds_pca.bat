@echo off
python cds_analysis.py --model tp_apc
python cds_analysis.py --model tp_apc --center
python cds_analysis.py --model tp_apc --center --standardize


python cds_analysis.py --model tw_apc
python cds_analysis.py --model tw_apc --center
python cds_analysis.py --model tw_apc --center --standardize

python cds_analysis.py --model complete
python cds_analysis.py --model complete --center
python cds_analysis.py --model complete --center --standardize
