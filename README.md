# Few-shot Anaphora Resolution in Scientific Protocols via Mixtures of In-Context Experts 

This repo contains code and data associated with EMNLP 2022 Findings paper [Few-shot Anaphora Resolution in Scientific Protocols via Mixtures of In-Context Experts](http://arxiv.org/abs/2210.03690)

### Installment

```
git clone https://github.com/nle18/mice.git
cd mice/src
conda env create -f environment.yml
conda activate gptj-gpu
export PYTHONPATH=.
```

### MICE Pipeline 

Prompt Generation: See example in scripts/prompt_generation.sh


Inference: See example in scripts/inference.sh 


Prompt Combination: See example in scripts/prompt_combination.sh 


Evaluation: See example in scripts/evaluation.sh  


