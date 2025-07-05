# Investigating Layer-Specific Vulnerability of LLMs to Adversarial Attacks

*Project for the **Computational Semantics for Natural Language Processing** (CSNLP) course at ETH Zürich in Spring Semester 2025.*

**Students**: Cagatay Gultekin, Fabio Giovanazzi, Adam Rahmoun, Tobias Kaiser.

**Initial project proposal**: [CSNL_Proposal__Final.pdf](./meta/CSNL_Proposal__Final.pdf)

**Midterm progress report**: [progress_report.pdf](./meta/progress_report.pdf)

**Final paper**: [Investigating_Layer_Specific_Vulnerability_of_LLMs_to_Adversarial_Attacks.pdf](./meta/Investigating_Layer_Specific_Vulnerability_of_LLMs_to_Adversarial_Attacks.pdf)

### Environment setup

Run these commands **in this order**:
```sh
conda create --name csnlp python=3.12 # python 3.12 works
conda activate csnlp # activate environment
pip install -r ./requirements.txt
```

### Training the model

The file [train.py](./train.py) is used to train the model using the regularization described in the paper. The script takes various parameters (get the full list with `--help`), though the relevant ones are:

- `--layers`: used to specify which layers are gradient-regularized
- `--learning_rate`: $\eta$ in the paper
- `--reg_lambda`: $\lambda$ in the paper
- `--max_length`: we used 512, as described in the paper
- `--batch_size`: we used 2, as described in the paper
- `--epochs`: we used 25000, which combined with a batch size of 2 makes for 50000 sequences
- `--model`: either "Qwen/Qwen2.5-1.5B-Instruct" or "microsoft/Phi-3-mini-4k-instruct"
- `--dataset`: this is by default "allenai/c4"

Sending `Ctrl+C` will run an inference on the partially trained model; use double `Ctrl+C` to exit.

### Evaluating perplexity

The file [evaluate_perplexity.py](./evaluate_perplexity.py) is used to compute perplexity of a model. This script also takes several parameters, the relevant ones are:

- `--max_length`: we used 512, as described in the paper
- `--batch_size`: we used 2, as described in the paper
- `--epochs`: we used 25000, which combined with a batch size of 2 makes for 50000 sequences
- `--model`: path to a local model for which to calculate perplexity, or either "Qwen/Qwen2.5-1.5B-Instruct" or "microsoft/Phi-3-mini-4k-instruct" to calculate perplexity of a pretrained model
- `--dataset`: this is by default "allenai/c4"

### Running GCG attacks

The Python notebook [evaluate_GCG_attacks.ipynb](./evaluate_GCG_attacks.ipynb) contains the code and the relevant documentation to run GCG attacks on the trained models and calculate various statistics.

### Raw results

All of the data results used in the paper can be found under the `.txt` files in the root of the repository. At the bottom of each file there is a **precise description** of how the results were obtained. To parse the results into a machine-readable format, use [results_to_latex.py](./results_to_latex.py), [results_to_latex_extended.py](./results_to_latex_extended.py) and [perplexity_to_latex.py](./perplexity_to_latex.py).

### Reimplemented model

The folder [reimplemented_model/](./reimplemented_model/) should be ignored, as it just contains a reimplementation of the tinyllama LLM that we made to learn more about its inner workings.
