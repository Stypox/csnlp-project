### Environment setup

Run these commands **in this order**:
```sh
conda create --name csnlp python=3.12 # python 3.12 works
conda activate csnlp # activate environment
conda install -c conda-forge -y cmake gxx # otherwise mismatching ABIs in rotary_emb
pip install -r ./requirements.txt
```