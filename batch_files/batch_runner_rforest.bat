@echo off

python synthetic_data_analysis.py --model miceforest 
python synthetic_data_analysis.py --model miceforest --center
python synthetic_data_analysis.py --model miceforest --center --standardize

python synthetic_data_analysis.py --model missforest
python synthetic_data_analysis.py --model missforest --center
python synthetic_data_analysis.py --model missforest --center --standardize