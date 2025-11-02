@echo off
python synthetic_data_analysis.py --model zero
python synthetic_data_analysis.py --model zero --center
python synthetic_data_analysis.py --model zero --center --standardize

python synthetic_data_analysis.py --model complete
python synthetic_data_analysis.py --model complete --center
python synthetic_data_analysis.py --model complete --center --standardize

python synthetic_data_analysis.py --model pca_weighted
python synthetic_data_analysis.py --model pca_weighted --center
python synthetic_data_analysis.py --model pca_weighted --center --standardize
