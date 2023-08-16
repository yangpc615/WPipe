# Environment initialization before runing wpipe

# requirements
pip install transformers==3.5.0
pip install tqdm
pip install pandas

# patches
transformers_package_root=`python -c "import transformers as _; print(_.__path__[0])"`
cp patch/modeling_bert.py ${transformers_package_root}/modeling_bert.py
cp patch/metric_init.py  ${transformers_package_root}/data/metrics/__init__.py

torch_package_root=`python -c "import torch as _; print(_.__path__[0])"`
cp patch/launch.py ${torch_package_root}/distributed/launch.py
