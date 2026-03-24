\documentclass[sigconf,nonacm]{acmart}

\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagestyle{plain}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{enumitem}
\usepackage{balance}
\usepackage{subcaption}
\usetikzlibrary{arrows.meta,positioning,fit}

\title{MASE-KD: Automated Knowledge Distillation Recovery for Pruned Neural Networks}

\author{Team Name}
\affiliation{%
  \institution{Imperial College London}
  \city{London}
  \country{United Kingdom}}

\begin{document}
\begin{abstract}
We present \textbf{MASE-KD}, a software-stream project built on top of MASE that automates a five-stage post-pruning recovery workflow (Fig.~\ref{fig:pipeline}): dense training, pruning, fine-tuning, knowledge distillation (KD), and KD followed by fine-tuning. On CIFAR-10 and CIFAR-100, the automated pipeline shows that KD becomes more valuable as sparsity increases and the task becomes harder. At 85\% sparsity on CIFAR-100, fine-tuning alone does not fully recover the dense baseline, whereas KD does.
\end{abstract}
\maketitle

\section{Purpose and System Scope}
Deep networks are often over-parameterised relative to deployment budgets. Magnitude pruning can compress them substantially, but aggressive pruning may severely damage accuracy \cite{han2015}. Our goal is therefore not just to prune a model, but to build a principled automated tool-flow that compares multiple recovery strategies under a unified interface. The current system targets three model families --- ResNet18, BERT, and YOLO.

\begin{figure}[t]
\centering
\begin{tikzpicture}[
  font=\small,
  node distance=3.5mm and 3.2mm,
  box/.style={
    draw,
    rounded corners=2pt,
    align=center,
    minimum height=7mm,
    text width=.145\columnwidth,
    inner sep=3pt,
    fill=gray!6
  },
  widebox/.style={
    draw,
    rounded corners=2pt,
    align=center,
    minimum height=7mm,
    text width=.86\columnwidth,
    inner sep=3pt,
    fill=gray!6
  },
  flow/.style={-Latex, thick}
]

\node[widebox] (cli) {Unified CLI + YAML config\\[-1pt]\footnotesize \texttt{run\_pipeline.py}};

% centered middle row
\node[box, below=5mm of cli] (c) {C\\FT};
\node[box, left=of c] (b) {B\\Prune};
\node[box, left=of b] (a) {A\\Dense};
\node[box, right=of c] (d) {D\\KD};
\node[box, right=of d] (e) {E\\KD+FT};

\node[widebox, below=5mm of c] (bottom) {Model-specific trainers + metric export\\[-1pt]\footnotesize ResNetKDTrainer, BertKDTrainer, YOLOKDRunner};

\draw[flow] (cli) -- (c);
\draw[flow] (a) -- (b);
\draw[flow] (b) -- (c);
\draw[flow] (c) -- (d);
\draw[flow] (d) -- (e);
\draw[flow] (c) -- (bottom);

\end{tikzpicture}
\caption{Compact view of the automated A--E workflow. Every stage emits a checkpoint and a metrics file, enabling reproducible cross-variant comparison.}
\label{fig:pipeline}
\end{figure}

\section{Architecture, Loss, and Metrics}
The system is organised around a shared KD core and a pipeline orchestrator. For a student with logits $z_s$, teacher logits $z_t$, labels $y$, temperature $T$, and mixing weight $\alpha$, we use the standard knowledge distillation objective \cite{hinton2015}
\begin{equation}
\mathcal{L}=(1-\alpha)\,\mathrm{CE}(z_s,y)+\alpha T^2\,\mathrm{KL}\!\left(\sigma\left(\tfrac{z_s}{T}\right)\,\middle\|\,\sigma\left(\tfrac{z_t}{T}\right)\right).
\end{equation}
Setting $\alpha=0$ recovers standard cross-entropy, so the same trainer implements dense training, fine-tuning, and KD.

For the experiment, we prioritise four metrics: test accuracy, sparsity ratio, non-zero parameter count, and the $D-C$ gap. Test accuracy is the main quality metric for the ResNet classification track; sparsity and non-zero parameters quantify compression; and $D-C$ directly measures whether KD helps more than ordinary fine-tuning after starting from the same pruned checkpoint.

\section{Key Design Decisions}
%\textbf{Self-distillation for ResNet18.} 
Instead of training a larger external teacher, the dense Step-A checkpoint is reused as the teacher for Steps D and E. This keeps the experiment budget tractable while preserving a meaningful dense target distribution.

\textbf{CIFAR-adapted stem.} Standard ResNet18 was originally introduced for ImageNet-scale inputs with a $7\times7$ stride-2 stem and max-pooling \cite{he2016}; this design is too aggressive for $32\times32$ CIFAR images. We replace it with a $3\times3$ stride-1 convolution and remove max-pooling.

\textbf{Learning-rate fix after pruning.} Early runs exposed catastrophic forgetting ($C < B$). Reducing the post-pruning fine-tune and KD learning rates from $10^{-2}$ to $10^{-3}$ stabilised recovery.

\textbf{Permanent pruning.} Global L1-unstructured pruning is made permanent with \texttt{prune.remove()}, ensuring that C/D/E all start from the same sparse checkpoint and making the $D-C$ comparison clean.

\section{ResNet18 Evaluation}
\subsection{Experimental Setup}
We evaluate a CIFAR-adapted ResNet18 \cite{he2016} on CIFAR-10 and CIFAR-100. Dense training uses SGD with momentum $0.9$, weight decay $5\times10^{-4}$, and cosine learning-rate scheduling. We test sparsity levels $s\in\{0.50,0.70,0.85\}$ with KD hyperparameters $\alpha=0.5$ and $T=4.0$. The reported results are from seed 0. At $s=0.85$, the model retains about $1.68$M non-zero parameters out of $11.17$M total, corresponding to roughly $6.6\times$ parameter compression.

\begin{table*}[t]
\caption{ResNet18 top-1 test accuracy across the A--E pipeline. Bold marks the best student within each dataset/sparsity setting.}
\label{tab:resnet-results}
\centering
\small
\begin{tabular}{llccccc}
\toprule
Dataset & Sparsity & A: Dense & B: Pruned & C: Pruned+FT & D: Pruned+KD & E: KD+FT \\
\midrule
\multirow{3}{*}{CIFAR-10}
& 0.50 & 0.9449 & 0.9447 & 0.9464 & \textbf{0.9472} & 0.9466 \\
& 0.70 & 0.9473 & 0.9297 & 0.9469 & 0.9464 & \textbf{0.9475} \\
& 0.85 & 0.9472 & 0.4362 & 0.9456 & 0.9464 & \textbf{0.9477} \\
\midrule
\multirow{3}{*}{CIFAR-100}
& 0.50 & 0.7699 & 0.7662 & 0.7697 & \textbf{0.7716} & 0.7709 \\
& 0.70 & 0.7654 & 0.7354 & 0.7645 & 0.7676 & \textbf{0.7683} \\
& 0.85 & 0.7686 & 0.5098 & 0.7599 & \textbf{0.7691} & 0.7687 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[t]
    \centering

    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figures/cifar_10_accuracy_vs_variant.png}
        \caption{CIFAR-10 accuracy across variants.}
        \label{fig:cifar10_variant}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figures/cifar_10_accuracy_vs_sparsity.png}
        \caption{CIFAR-10 recovery delta versus sparsity.}
        \label{fig:cifar10_sparsity}
    \end{subfigure}

    \vspace{0.5em}

    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figures/cifar_100_accuracy_vs_variant.png}
        \caption{CIFAR-100 accuracy across variants.}
        \label{fig:cifar100_variant}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figures/cifar_100_accuracy_vs_sparsity.png}
        \caption{CIFAR-100 recovery delta versus sparsity.}
        \label{fig:cifar100_sparsity}
    \end{subfigure}

    \caption{Comparison of ResNet18 performance across datasets, variants, and sparsity settings. The subfigures summarise absolute accuracy and recovery trends after pruning, fine-tuning, and knowledge distillation.}
    \label{fig:resnet_2x2_results}
\end{figure}

\subsection{Results and Discussion}
Three patterns are immediately apparent from Fig.~\ref{fig:resnet_2x2_results} and Table~\ref{tab:resnet-results}. First, CIFAR-10 at 50\% sparsity is a \emph{free-lunch zone}: pruning barely changes accuracy (0.9447 vs. 0.9449), indicating heavy over-parameterisation. Second, at 85\% sparsity the unrecovered student collapses on both datasets, but recovery remains strong: on CIFAR-10, E reaches 0.9477 and slightly exceeds the dense baseline; on CIFAR-100, D reaches 0.7691 and is the only variant that beats the dense model. Third, the KD benefit scales with difficulty. The $D-C$ gap is only $+0.0008$ on CIFAR-10 at 85\% sparsity, but rises to $+0.0092$ on CIFAR-100 at the same sparsity, showing that teacher soft labels are most valuable when pruning damage is severe and hard labels alone are not sufficiently informative.

% \section{Testing and Engineering Validation}
% The implementation follows a pass-based layout under \texttt{src/mase\_kd} with coursework tests isolated in \texttt{cw/}. For the ResNet path, unit tests target pruning, export, and config logic; integration smoke tests validate one-epoch training, frozen-teacher behaviour, checkpoint save/load consistency, and the end-to-end B$\rightarrow$C recovery flow. This testing strategy directly supports the engineering requirements of the coursework: correctness of the automated tool-flow, modularity, and reproducible artifact generation.

\section{BERT}


\section{YOLOv8 Classification}

Unlike the ResNet track, which uses self-distillation from a dense checkpoint of the same architecture, the YOLOv8 track evaluates \emph{cross-architecture} KD: a large teacher (\texttt{yolov8x-cls}, 71.4\,M parameters) distils into a smaller student (\texttt{yolov8m-cls}, 18.9\,M parameters)---a ${\sim}3.8\times$ capacity gap. Both models are loaded via MASE's \texttt{MaseYoloClassificationModel} wrapper and share the same KD loss (Eq.\,1). We conduct a systematic $5\times5$ grid search over $\alpha\in\{0.3,0.5,0.7,0.9,1.0\}$ and $T\in\{1,2,4,8,16\}$ to map the full KD hyperparameter landscape.

\subsection{YOLO-Specific Design Decisions}

\textbf{Raw-logits extraction via train-mode trick.}
The Ultralytics \texttt{Classify} head returns \texttt{x.softmax(1)} in eval mode. Applying $\sigma(\text{softmax}/T)$ downstream would produce a double-softmax, destroying the dark-knowledge signal. The teacher is therefore temporarily switched to \texttt{train()} inside a \texttt{torch.no\_grad()} block so the head returns raw logits.

\textbf{Strict shape assertion.}
\texttt{\_align\_logits} raises a \texttt{ValueError} on any student--teacher shape mismatch rather than silently truncating, catching the Classify head's mode-dependent output format (tuple vs.\ single tensor) immediately.

\textbf{Epoch-based best-model checkpointing.}
\texttt{train()} saves the student's \texttt{state\_dict()} whenever validation top-1 accuracy improves; \texttt{strict=False} is required during \texttt{load\_state\_dict} because MASE's pruning pass registers sparsity masks as non-persistent buffers.

\subsection{Experimental Setup}

The teacher is fine-tuned on each target dataset; the student starts from ImageNet-pretrained weights (no dataset-specific pre-training). All 26~runs per dataset (1~CE-only baseline + 25~KD configurations) share: AdamW with $\text{lr}=5\times10^{-4}$, weight decay 0.05, batch size 128, 60~epochs, seed~42. Four conditions are evaluated per run: teacher, untrained student, CE-only fine-tuned student, and KD-distilled student.

\subsection{Results and Discussion}

\begin{table}[t]
\caption{YOLOv8 headline results: best KD configuration vs.\ CE-only fine-tuning on CIFAR-100 and CIFAR-10.}
\label{tab:yolo-headline}
\centering
\small
\begin{tabular}{lrrl}
\toprule
Condition & \multicolumn{1}{c}{CIFAR-100} & \multicolumn{1}{c}{CIFAR-10} & Best config \\
\midrule
Teacher (yolov8x-cls) & 68.33\% & 91.25\% & --- \\
Untrained student     &  0.93\% & 10.64\% & --- \\
CE-only fine-tuned    & 50.99\% & 83.04\% & $\alpha{=}0$ \\
\textbf{Best KD}      & \textbf{60.46\%} & \textbf{84.93\%} & $\alpha{=}1.0,T{=}16$ / $\alpha{=}0.9,T{=}8$ \\
\midrule
KD gain over CE       & \textbf{+9.47\,pp} & \textbf{+1.88\,pp} & \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:yolo-headline} and Fig.~\ref{fig:yolo-bars} summarise the headline comparison. On CIFAR-100, KD closes nearly half the remaining teacher--student gap ($+9.47$\,pp over CE-only), whereas on CIFAR-10 the gain is a more modest $+1.88$\,pp. This mirrors the ResNet finding: KD is most valuable when the classification task is harder and soft targets carry richer inter-class structure.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/bar_4condition.png}
\caption{Four-condition comparison for YOLOv8-cls on CIFAR-100 and CIFAR-10. The $\Delta$ annotation shows the KD gain over CE-only fine-tuning.}
\label{fig:yolo-bars}
\end{figure}

Fig.~\ref{fig:yolo-heatmap} presents the full $5\times5$ grid of KD gains. On CIFAR-100, \emph{every} KD configuration outperforms CE-only (minimum $+0.48$\,pp at $\alpha{=}0.5, T{=}1$; maximum $+9.47$\,pp at $\alpha{=}1.0, T{=}16$). Higher temperatures and higher $\alpha$ values generally produce larger gains, confirming Hinton et~al.'s prediction that soft targets are most informative when temperature is high enough to spread probability mass across semantically related classes. On CIFAR-10, 24 of 25 configurations improve over CE-only; the single exception ($\alpha{=}0.3, T{=}1$, $-0.16$\,pp) is consistent with $T{=}1$ being a degenerate case where soft targets approximate hard labels.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/heatmap_kd_gain.png}
\caption{KD gain (pp) over CE-only fine-tuning across the $\alpha\times T$ grid. Left: CIFAR-100; right: CIFAR-10. Darker green indicates larger improvement.}
\label{fig:yolo-heatmap}
\end{figure}

Fig.~\ref{fig:yolo-lines} reveals the temperature dynamics more clearly. On CIFAR-100, accuracy climbs monotonically with $T$ for all $\alpha$ values, with $T{=}16$ consistently best. On CIFAR-10, the curves are flatter and peak around $T{=}2$--$8$, reflecting the smaller dark-knowledge budget of a 10-class problem. The key finding is that \textbf{KD benefit scales with classification complexity}: the improvement is roughly $5\times$ larger on CIFAR-100 than on CIFAR-10, consistent with the theory that dark knowledge is richer when there are more inter-class relationships to encode.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/line_top1_vs_temp.png}
\caption{KD top-1 accuracy vs.\ temperature for each $\alpha$. Dashed lines show teacher and CE-only baselines. Left: CIFAR-100; right: CIFAR-10.}
\label{fig:yolo-lines}
\end{figure}

\section{GPT-2}
\section{Conclusion}
The ResNet18 track already demonstrates that MASE-KD is more than a simple wrapper around pruning and training scripts: it is an automated experimental framework that compares multiple recovery strategies under a shared interface. For image classification, the main conclusion is clear: \textbf{KD becomes more useful as pruning gets harsher and the task gets harder}. On CIFAR-100 at 85\% sparsity, fine-tuning alone cannot fully recover the dense baseline, but KD can. This section can therefore stand as the completed vision-classification contribution while the BERT and YOLO sections are integrated later.

\begin{thebibliography}{9}
\bibitem{hinton2015}
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.
Distilling the Knowledge in a Neural Network.
\emph{NeurIPS Deep Learning Workshop}.

\bibitem{he2016}
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.
Deep Residual Learning for Image Recognition.
In \emph{Proceedings of CVPR}.

\bibitem{han2015}
Song Han, Jeff Pool, John Tran, and William Dally. 2015.
Learning both Weights and Connections for Efficient Neural Networks.
In \emph{Advances in Neural Information Processing Systems}.
\end{thebibliography}

\balance
\end{document}
