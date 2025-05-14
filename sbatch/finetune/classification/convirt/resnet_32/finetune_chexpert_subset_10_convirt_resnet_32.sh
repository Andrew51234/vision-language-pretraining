#!/bin/bash
#SBATCH --job-name=CheXpert-subset-10-convirt-resnet-32
#SBATCH --output=logs/sbatch/CheXpert-subset-10-convirt-resnet-32-%A.out
#SBATCH --error=logs/sbatch/CheXpert-subset-10-convirt-resnet-32-%A.err
#SBATCH --mail-user=viet.x.pham@tum.de
#SBATCH --mail-type=ALL
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G


# change directory to the one containing the script
cd /vol/aimspace/users/phvi/GitLab/vlp

# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you
# always run this script from a clean terminal

source deactivate

# If the following does not work, try 'source activate <env-name>'
source activate vlp-dev

# Run the program
# Use srun to run on a slurm node
srun python src/finetune_chexpert.py +experiment=finetune/classification/convirt/resnet_32/finetune_chexpert_subset_10_convirt_resnet_32


Here's our experiments directory content:
For classification:
there is convirt with the following dirs:
-default: Containing no freeze, default, subset 10 no freeze and subset 10 default
- no flip: with same files
- resnet32, resnet 128, resnet 256, vit32, vit128, vit256, imagenet: Containing default and no freeze and same for subset 1 and 10
Then outside:
 imagenet: Containing default and no freeze and same for subset 1 and 10
maskvlm 03 and 06: default and no freeze and same for subset 1 and 10
random: default and freeze and 1 and 10 subsets

For maskvlm we have the following for subset 1 and 10:
default, freeze bert, loss weights, n sentences, optimized


For the retrieval we have also convirt and maskvlm:
Convirt has resnet 3 sentences, resnet32, resnet128, resnet 256, vit32, vit128, vit256, imagenet
Maskvlm has default, freeze bert, loss weights, n sentences, optimized, random

Now return me only the modified subsections in this latex:
Our experiments aim to dissect the contributions of various training modifications for ConVIRT and MaskVLM. The experiments, summarized in Table~\ref{tab:experiment_settings}, systematically vary key factors such as batch size, loss type, the freezing of the text encoder, and the number of sentences sampled from each report. The main motivations behind these variations are as follows:

\begin{itemize}
    \item \textbf{Flipping Augmentation:} Only the ConVIRT Default experiment applies horizontal flipping, as this augmentation could interfere with the semantic interpretation of directional keywords (e.g., “left” and “right”) present in medical reports.
    
    \item \textbf{Batch Size Variation:} Contrastive learning benefits from larger batch sizes due to the increased number of negative examples. For ConVIRT, experiments with batch sizes of 32, 128, and 256 were performed. For MaskVLM, memory limitations restricted large-batch experiments, leading to two settings with 32 and 64.
    
    \item \textbf{Text Encoder Freezing:} Inspired by ConVIRT’s strategy, some MaskVLM configurations freeze the first six layers of the BERT text encoder. This aims to determine if freezing lower layers stabilizes training or improves performance. Further motivated was this change since we encountered early overfitting when training the first versions of MaskVLM.
    
    \item \textbf{Sentence Sampling:} While most experiments use one sentence per report, additional experiments sample three sentences to provide richer textual context. This is intended to mitigate the risk of selecting a non-meaningful sentence.
    
    
    \item \textbf{Loss Function Combinations:} Most MaskVLM experiments use a combination of Contrastive (C) and Reconstruction (R) losses. A dedicated MaskVLM experiment using only reconstruction loss (R) isolates the impact of the reconstruction component on model performance.
\end{itemize}

To summarize our experimental configurations, Table~\ref{tab:experiment_settings} shows the details of all ConVIRT and MaskVLM runs, including the settings for text encoder freezing, image encoder choice, batch sizes, loss types (with "C" for Contrastive and "R" for Reconstruction), and the number of sentences used from the report, noting that only the ConVIRT Default configuration applies horizontal flipping.

\begin{table}[h]
    \centering
    \caption{Comparison of experimental settings for MaskVLM and ConVIRT. \textit{C}=Contrastive Loss, \textit{R}=Reconstruction Loss. Only ConVIRT Default applies flipping augmentation. \textit{frozen} describes the number of the first layers in the text encoder which are frozen for pretraining}
    \begin{tabular}{|lccccc|}
        \hline
        \textbf{Experiment} & \textbf{frozen} & \textbf{Img. Encoder }& \textbf{BS} & \textbf{Loss} & \textbf{\#Sent.} \\
        \hline
        ConVIRT Default & 6 & ResNet50 & 32 & C & 1 \\
        ConVIRT (BS=32) & 6 & ResNet50 & 32 & C & 1 \\
        ConVIRT (BS=128) & 6 & ResNet50 & 128 & C & 1 \\
        ConVIRT (BS=256) & 6 & ResNet50 & 256 & C & 1 \\
        ConVIRT ViT (BS=32) & 6 & ViT  & 32 & C & 1 \\
        ConVIRT ViT (BS=128) & 6 & ViT  & 128 & C & 1 \\
        ConVIRT ViT (BS=256) & 6 & ViT  & 256 & C & 1 \\
        ConVIRT 3 Sentences & 6 & ResNet50 & 256 & C & 3 \\
        MaskVLM Default & 0 & ViT & 32 & C + R & 1 \\
        MaskVLM Freeze BERT & 6 & ViT & 32 & C + R & 1 \\
        MaskVLM optimized &  4 & ViT & 64 & C + R & 2 \\
        MaskVLM Reconstruction Only & 0 & ViT & 32 & R & 1 \\
        MaskVLM 3 Sentences & 0 & ViT & 32 & C + R & 3 \\
        \hline
    \end{tabular}
    \label{tab:experiment_settings}
\end{table}
\todo{check if correct}

\subsection{Evaluating on Classification}
\todo{}

\subsection{Evaluating on Retrieval Tasks}
\todo{}
