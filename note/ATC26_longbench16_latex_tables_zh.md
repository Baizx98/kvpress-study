# ATC26 LongBench-16 LaTeX Table

说明：以下 LaTeX 表格只使用 `fraction=1.0` 的 full 结果，已过滤 smoking test 的 `fraction=0.01` 行；同一 `model/data_dir/method/compression_ratio` 若存在重复记录，则保留最后一次结果。三种模型被合并到同一个大表中，不同模型之间用 `\multicolumn` 行隔开。表头第一层为 LongBench 任务类型，第二层为具体数据集名称；由于 16 个数据集列较多，数据集名称使用 `\rotatebox` 旋转显示。

LongBench-16 任务类型划分：

- Single-Doc QA：NarrativeQA、Qasper、MultiFieldQA
- Multi-Doc QA：HotpotQA、2WikiMQA、MuSiQue
- Summarization：GovReport、QMSum、MultiNews
- Few-shot：TriviaQA、SAMSum、TREC
- Synthetic：PassageCnt、PassageRet
- Code：LCC、RepoBench-P

LaTeX 依赖建议：`\usepackage{booktabs}`、`\usepackage{graphicx}`。如果目标模板不允许 `table*`，可以把环境改为 `sidewaystable` 或拆成按模型分表。

```latex
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.0pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllrrrrrrrrrrrrrrrrr}
\toprule
Model & Ratio & Method & \multicolumn{3}{c}{Single-Doc QA} & \multicolumn{3}{c}{Multi-Doc QA} & \multicolumn{3}{c}{Summarization} & \multicolumn{3}{c}{Few-shot} & \multicolumn{2}{c}{Synthetic} & \multicolumn{2}{c}{Code} & Avg. \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}\cmidrule(lr){16-17}\cmidrule(lr){18-19}
 &  &  & \rotatebox{60}{NarrativeQA} & \rotatebox{60}{Qasper} & \rotatebox{60}{MultiFieldQA} & \rotatebox{60}{HotpotQA} & \rotatebox{60}{2WikiMQA} & \rotatebox{60}{MuSiQue} & \rotatebox{60}{GovReport} & \rotatebox{60}{QMSum} & \rotatebox{60}{MultiNews} & \rotatebox{60}{TriviaQA} & \rotatebox{60}{SAMSum} & \rotatebox{60}{TREC} & \rotatebox{60}{PassageCnt} & \rotatebox{60}{PassageRet} & \rotatebox{60}{LCC} & \rotatebox{60}{RepoBench-P} & Avg. \\
\midrule
\multicolumn{20}{c}{\textbf{Llama-3.1-8B-Instruct}} \\
\midrule
 & 0.3 & BlockWisePress & 30.07 & \textbf{48.09} & \textbf{56.20} & 58.82 & 50.04 & 32.69 & 33.99 & 24.66 & 26.29 & \textbf{92.23} & 39.59 & 28.50 & 11.05 & \textbf{100.00} & 53.04 & \textbf{49.23} & 45.91 \\
 &  & SnapKV & 30.69 & 46.77 & 55.48 & \textbf{59.43} & \textbf{51.10} & 32.39 & 33.71 & 24.75 & \textbf{26.35} & 91.41 & \textbf{41.04} & \textbf{31.50} & \textbf{11.20} & \textbf{100.00} & \textbf{53.87} & 47.52 & \textbf{46.08} \\
 &  & ChunkKV & \textbf{30.76} & 46.83 & 55.65 & 58.31 & 50.10 & \textbf{33.93} & \textbf{34.19} & \textbf{24.92} & 26.31 & 91.41 & 40.47 & 28.50 & 11.15 & \textbf{100.00} & 52.46 & 47.94 & 45.81 \\
\addlinespace[1pt]
 & 0.4 & BlockWisePress & 29.58 & 47.87 & \textbf{55.86} & 58.96 & 49.05 & 31.67 & \textbf{33.71} & 24.56 & 26.08 & 91.54 & 38.87 & 24.50 & 11.13 & \textbf{100.00} & 52.92 & \textbf{48.39} & 45.29 \\
 &  & SnapKV & 30.21 & 47.11 & 54.83 & \textbf{59.19} & \textbf{51.71} & \textbf{33.31} & 33.50 & 24.66 & 26.12 & \textbf{91.88} & \textbf{40.69} & \textbf{37.50} & 10.65 & \textbf{100.00} & \textbf{53.04} & 47.25 & \textbf{46.35} \\
 &  & ChunkKV & \textbf{30.61} & \textbf{48.08} & 55.72 & 59.16 & 51.54 & 32.77 & 33.62 & \textbf{24.79} & \textbf{26.53} & 91.41 & 40.46 & 24.50 & \textbf{12.55} & \textbf{100.00} & 52.44 & 48.33 & 45.78 \\
\addlinespace[1pt]
 & 0.5 & BlockWisePress & 29.38 & 47.34 & 54.59 & 58.23 & 49.52 & 31.30 & 32.83 & 24.47 & 25.55 & 91.72 & 39.43 & 23.00 & 11.05 & \textbf{100.00} & 52.02 & \textbf{48.93} & 44.96 \\
 &  & SnapKV & \textbf{30.79} & 46.22 & \textbf{55.49} & 59.14 & \textbf{50.98} & \textbf{32.84} & 32.64 & 24.51 & \textbf{25.66} & \textbf{91.88} & \textbf{40.56} & \textbf{36.50} & \textbf{11.65} & \textbf{100.00} & \textbf{53.09} & 47.17 & \textbf{46.20} \\
 &  & ChunkKV & 30.44 & \textbf{47.46} & 54.68 & \textbf{59.16} & 49.77 & 32.38 & \textbf{33.43} & \textbf{24.81} & 25.64 & 91.43 & 39.67 & 24.00 & 10.55 & \textbf{100.00} & 50.81 & 48.54 & 45.17 \\
\addlinespace[1pt]
 & 0.6 & BlockWisePress & 27.38 & \textbf{47.97} & 51.83 & 57.51 & 49.62 & 31.56 & 31.81 & 23.85 & \textbf{24.73} & \textbf{91.98} & 38.00 & 24.00 & 7.50 & 99.50 & 51.22 & \textbf{50.27} & 44.30 \\
 &  & SnapKV & \textbf{30.60} & 45.46 & \textbf{55.14} & \textbf{59.25} & \textbf{51.15} & \textbf{32.94} & 31.41 & \textbf{24.86} & 24.64 & 91.54 & 39.85 & \textbf{36.50} & 11.70 & \textbf{100.00} & \textbf{53.28} & 47.15 & \textbf{45.97} \\
 &  & ChunkKV & 30.20 & 47.08 & 54.96 & 58.33 & 48.70 & 32.87 & \textbf{32.35} & 24.39 & 24.52 & \textbf{91.98} & \textbf{40.39} & 26.50 & \textbf{12.05} & \textbf{100.00} & 51.18 & 48.86 & 45.27 \\
\addlinespace[1pt]
 & 0.7 & BlockWisePress & 26.99 & \textbf{46.53} & 50.43 & 57.97 & 49.02 & 31.01 & 30.56 & 23.96 & \textbf{24.35} & 91.48 & 36.60 & 22.00 & 6.54 & 99.50 & 51.08 & \textbf{50.48} & 43.66 \\
 &  & SnapKV & \textbf{30.03} & 46.04 & \textbf{55.84} & \textbf{59.04} & \textbf{50.92} & \textbf{32.56} & 30.47 & \textbf{24.49} & 23.87 & \textbf{92.08} & \textbf{40.66} & \textbf{41.00} & 11.63 & \textbf{100.00} & \textbf{53.32} & 47.52 & \textbf{46.22} \\
 &  & ChunkKV & 29.63 & 46.35 & 55.66 & 58.87 & 47.79 & 32.39 & \textbf{31.08} & 24.34 & 23.46 & 91.68 & 38.74 & 24.00 & \textbf{12.55} & 99.50 & 50.83 & 49.13 & 44.75 \\
\midrule
\multicolumn{20}{c}{\textbf{Mistral-7B-Instruct-v0.3}} \\
\midrule
 & 0.3 & BlockWisePress & 26.95 & 38.57 & 51.33 & \textbf{48.97} & \textbf{39.46} & 27.84 & 33.82 & 25.34 & \textbf{26.50} & 86.02 & \textbf{24.06} & \textbf{50.73} & 4.73 & 98.00 & \textbf{51.57} & 56.46 & \textbf{43.15} \\
 &  & SnapKV & 27.95 & 38.44 & 51.01 & 48.75 & 38.80 & 28.11 & 33.85 & \textbf{25.57} & 26.08 & \textbf{86.49} & 22.79 & 49.63 & 5.17 & 98.00 & 51.14 & \textbf{56.51} & 43.02 \\
 &  & ChunkKV & \textbf{28.07} & \textbf{39.41} & \textbf{51.70} & 48.60 & 38.40 & \textbf{28.93} & \textbf{33.99} & 25.29 & 26.01 & 85.42 & 22.58 & 48.92 & \textbf{6.12} & \textbf{98.50} & 51.54 & 56.03 & 43.09 \\
\addlinespace[1pt]
 & 0.4 & BlockWisePress & 26.27 & 38.60 & 51.67 & 48.54 & \textbf{39.62} & 27.61 & 33.51 & 24.85 & 25.98 & \textbf{86.52} & \textbf{24.57} & \textbf{51.76} & 5.25 & 98.00 & \textbf{52.51} & 56.33 & \textbf{43.22} \\
 &  & SnapKV & 27.42 & 38.55 & 51.21 & 48.71 & 38.89 & \textbf{28.41} & 33.23 & \textbf{25.26} & \textbf{26.07} & 85.99 & 22.69 & 49.12 & \textbf{5.77} & \textbf{98.50} & 52.50 & \textbf{56.82} & 43.07 \\
 &  & ChunkKV & \textbf{27.54} & \textbf{39.61} & \textbf{51.70} & \textbf{48.90} & 37.18 & 28.34 & \textbf{33.63} & 24.86 & 25.83 & 85.96 & 22.80 & 48.10 & 4.92 & 98.00 & 51.44 & 56.47 & 42.83 \\
\addlinespace[1pt]
 & 0.5 & BlockWisePress & 25.65 & 37.37 & 50.51 & 47.82 & 36.35 & 27.19 & \textbf{33.14} & 24.06 & 25.44 & \textbf{86.92} & \textbf{24.74} & \textbf{51.42} & 4.53 & \textbf{98.50} & 52.23 & 56.46 & 42.65 \\
 &  & SnapKV & 27.23 & 38.47 & \textbf{51.14} & 48.01 & \textbf{38.74} & \textbf{27.99} & 32.60 & \textbf{25.24} & \textbf{25.48} & 86.56 & 23.25 & 49.63 & 4.85 & \textbf{98.50} & 52.52 & \textbf{57.72} & \textbf{43.00} \\
 &  & ChunkKV & \textbf{27.52} & \textbf{39.87} & 50.67 & \textbf{48.34} & 37.41 & 27.95 & 32.95 & 24.94 & \textbf{25.48} & 86.23 & 24.23 & 45.67 & \textbf{5.27} & 98.00 & \textbf{52.78} & 57.24 & 42.78 \\
\addlinespace[1pt]
 & 0.6 & BlockWisePress & 25.36 & \textbf{38.13} & 50.45 & \textbf{48.59} & 36.01 & 25.16 & \textbf{31.97} & 23.63 & \textbf{25.08} & \textbf{87.66} & \textbf{26.26} & \textbf{51.74} & 4.03 & \textbf{98.50} & 52.09 & 57.19 & 42.62 \\
 &  & SnapKV & 27.11 & 37.06 & \textbf{51.02} & 48.58 & \textbf{38.28} & 27.37 & 31.51 & \textbf{24.88} & 24.62 & 87.00 & 23.58 & 48.38 & 4.01 & \textbf{98.50} & \textbf{53.38} & \textbf{58.03} & \textbf{42.71} \\
 &  & ChunkKV & \textbf{27.63} & 37.68 & 50.40 & 47.42 & 36.38 & \textbf{27.83} & 31.69 & 24.63 & 24.65 & 86.26 & 25.91 & 40.89 & \textbf{5.26} & \textbf{98.50} & 52.88 & 57.40 & 42.21 \\
\addlinespace[1pt]
 & 0.7 & BlockWisePress & 25.84 & 35.74 & 49.62 & \textbf{48.31} & 34.74 & 25.23 & 30.51 & 23.11 & \textbf{24.28} & \textbf{87.85} & \textbf{24.98} & \textbf{51.04} & 2.84 & 97.50 & 51.79 & 56.78 & 41.88 \\
 &  & SnapKV & 27.00 & 35.82 & \textbf{50.69} & 47.69 & \textbf{37.67} & 27.70 & 30.28 & \textbf{24.98} & 23.95 & 86.94 & 24.66 & 48.50 & 4.68 & \textbf{98.50} & 53.34 & \textbf{57.99} & \textbf{42.52} \\
 &  & ChunkKV & \textbf{27.58} & \textbf{36.39} & 50.20 & 47.48 & 35.69 & \textbf{28.00} & \textbf{30.89} & 24.38 & 23.47 & 86.76 & 24.95 & 36.75 & \textbf{5.55} & \textbf{98.50} & \textbf{54.07} & 57.26 & 41.74 \\
\midrule
\multicolumn{20}{c}{\textbf{Qwen3-8B}} \\
\midrule
 & 0.3 & BlockWisePress & \textbf{29.73} & \textbf{44.20} & 53.95 & \textbf{62.62} & 48.75 & 33.79 & \textbf{33.70} & \textbf{24.45} & \textbf{24.23} & 90.26 & 40.24 & 39.00 & \textbf{9.50} & \textbf{94.54} & 66.78 & 62.25 & 47.37 \\
 &  & SnapKV & 28.92 & 44.17 & \textbf{55.27} & 62.05 & \textbf{49.58} & \textbf{35.88} & 33.10 & 24.19 & 24.21 & \textbf{90.46} & 39.89 & \textbf{41.50} & \textbf{9.50} & 92.43 & \textbf{67.12} & \textbf{62.52} & \textbf{47.55} \\
 &  & ChunkKV & 28.35 & 43.42 & 54.94 & 62.30 & 48.62 & 35.78 & 33.20 & 24.43 & 24.11 & 90.06 & \textbf{40.39} & 40.00 & 8.50 & 91.35 & 66.91 & 62.51 & 47.18 \\
\addlinespace[1pt]
 & 0.4 & BlockWisePress & \textbf{30.79} & 44.50 & 52.90 & 62.28 & \textbf{50.61} & 35.00 & \textbf{33.03} & 23.86 & \textbf{23.93} & 90.16 & \textbf{40.90} & 35.50 & 9.00 & \textbf{95.79} & 66.85 & 62.88 & \textbf{47.37} \\
 &  & SnapKV & 29.03 & 43.88 & \textbf{54.56} & 62.05 & 48.88 & \textbf{35.57} & 32.78 & 24.34 & 23.78 & 89.96 & 40.68 & 40.00 & \textbf{9.50} & 91.77 & \textbf{67.39} & 62.72 & 47.31 \\
 &  & ChunkKV & 28.15 & \textbf{44.54} & 54.44 & \textbf{62.39} & 49.05 & 34.31 & 32.72 & \textbf{24.71} & 23.54 & \textbf{90.21} & 40.14 & \textbf{41.50} & 7.50 & 92.02 & 67.20 & \textbf{62.94} & 47.21 \\
\addlinespace[1pt]
 & 0.5 & BlockWisePress & \textbf{29.42} & 43.58 & 52.20 & \textbf{62.36} & 49.60 & 33.77 & \textbf{32.85} & 23.73 & \textbf{23.48} & \textbf{90.24} & 40.24 & 34.00 & \textbf{9.50} & \textbf{97.08} & 65.70 & 62.85 & 46.91 \\
 &  & SnapKV & 29.07 & 43.16 & \textbf{54.87} & 61.63 & 49.28 & \textbf{35.90} & 32.25 & 24.19 & 23.21 & 89.96 & \textbf{40.62} & 39.50 & 8.50 & 92.39 & \textbf{68.15} & \textbf{63.36} & \textbf{47.25} \\
 &  & ChunkKV & 28.35 & \textbf{43.62} & 54.58 & 61.69 & \textbf{50.08} & 35.10 & 32.50 & \textbf{24.42} & 22.86 & 90.18 & 39.90 & \textbf{40.00} & 7.00 & 91.60 & 67.29 & 63.09 & 47.02 \\
\addlinespace[1pt]
 & 0.6 & BlockWisePress & 27.99 & 42.84 & 52.30 & 62.11 & 47.76 & 33.09 & \textbf{32.54} & 23.39 & \textbf{22.92} & \textbf{90.24} & 39.82 & 32.00 & 8.05 & \textbf{96.85} & 66.22 & 63.89 & 46.38 \\
 &  & SnapKV & \textbf{28.61} & 43.17 & 53.94 & 61.64 & \textbf{49.78} & \textbf{35.83} & 31.79 & \textbf{24.28} & 22.43 & 90.21 & \textbf{40.85} & \textbf{38.00} & \textbf{9.00} & 93.04 & \textbf{67.49} & \textbf{64.21} & \textbf{47.14} \\
 &  & ChunkKV & 27.98 & \textbf{43.72} & \textbf{54.44} & \textbf{62.17} & 49.56 & 34.98 & 31.83 & 23.69 & 22.39 & 89.93 & 40.62 & 37.00 & 7.05 & 92.56 & 67.42 & 63.91 & 46.83 \\
\addlinespace[1pt]
 & 0.7 & BlockWisePress & 29.47 & 42.39 & 52.54 & \textbf{62.18} & 47.12 & 32.46 & \textbf{31.68} & 23.30 & \textbf{21.78} & 88.92 & 39.88 & 29.50 & 7.00 & \textbf{97.10} & 65.77 & 64.20 & 45.96 \\
 &  & SnapKV & 28.12 & 42.19 & 52.04 & 62.13 & \textbf{49.41} & \textbf{35.58} & 30.73 & \textbf{24.26} & 21.35 & \textbf{90.21} & \textbf{41.14} & \textbf{34.75} & \textbf{7.55} & 93.10 & \textbf{67.80} & \textbf{64.93} & \textbf{46.58} \\
 &  & ChunkKV & \textbf{29.62} & \textbf{42.52} & \textbf{54.35} & 61.99 & 48.61 & 34.77 & 31.10 & 23.21 & 21.49 & \textbf{90.21} & 40.38 & 33.50 & 7.00 & 91.99 & 67.56 & 64.22 & 46.41 \\
\bottomrule
\end{tabular}%
}
\caption{ATC26 LongBench-16 full results across three models. Columns are grouped by LongBench task type, and dataset names are rotated to fit the paper width. Higher score is better. For each model and compression ratio, the best result among BlockWisePress, SnapKV, and ChunkKV is bolded.}
\label{tab:atc26-longbench16-all-models}
\end{table*}
```
