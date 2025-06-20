import math

f = open("Perplexity_Results.txt").read()
f = f.split("Qwen\n\n")[1]
qwen, phi3 = f.split("\n\nPhi3\n\n")
phi3 = phi3.split("\n\nImplementation notes:")[0]

qwen_out, phi3_out = [], []
for block, name, out in [(qwen, "Qwen", qwen_out), (phi3, "Phi-3", phi3_out)]:
    for line in block.split("\n\n"):
        line, perplexity = line.split("\nFinal Perplexity: ")
        perplexity = float(perplexity)
        if "Original model" in line:
            out.append((0, "Orig", perplexity))
        elif "_layers_" not in line:
            out.append((0, "None", perplexity))
        else:
            assert "lr_1e-4" in line
            reg, layers = line.split("_reg_")[1].split("_lr_1e-4_layers_")
            reg = float(reg)
            if "0,1," in layers:
                layers = "early"
            elif "12,13," in layers:
                layers = "middle"
            else:
                assert "22,23," in layers
                layers = "late"
            out.append((reg, layers, perplexity))
assert len(qwen_out) == 20
assert len(phi3_out) == 20

def format_perplexity(p):
    if p < 100:
        return f"{p:.1f}"
    else:
        return f"{int(p)}*"

print(r"""\begin{table}[h]
\centering
\begin{tabular}{|c c|c c|}
\hline
\multicolumn{2}{|c|}{\textbf{Regularization}} & \multicolumn{2}{c|}{\textbf{Perplexity}} \\
$\lambda$ & $I$ & \textbf{Qwen} & \textbf{Phi-3} \\""")
for i, ((q_reg, q_layers, q_perplexity), (p_reg, p_layers, p_perplexity)) in enumerate(zip(qwen_out, phi3_out)):
    assert q_reg == p_reg
    assert p_layers == q_layers
    if i == 0:
        continue
    elif i == 1:
        print(r"""\hline \multicolumn{2}{|c|}{none} """, end="")
    else:
        if i % 3 == 2:
            print(rf"""\hline \multirow{{3}}{{*}}{{$10^{{{int(math.log10(q_reg))}}}$}}""", end="")
        print(rf"""& {q_layers} """, end="")
    print(rf"""& {format_perplexity(q_perplexity)} & {format_perplexity(p_perplexity)} \\""")
print(r"""\hline
\end{tabular}
\caption{Perplexity of fine-tuned models, lower is better}
\label{tab:perplexity}
\end{table}""")