# Demonstrating limitations of Natural Language Autoencoders in small open weight models

## Why did I choose this problem?
I haven’t worked extensively with autoencoders before, though I have a rough high-level understanding of SAEs: how they work and some of their limitations. That said, I found the idea of NLAs quite interesting, and this project felt like a good way to better understand what is going on and build first order intuition about the limitations of NLAs.

## My understanding of NLAs
It comprises of two components 
1. Activation verbaliser (AV): Take a token representation $h_l \in \mathbb{R}^d$ and autoregressively generate the explanation $z$. 
2. Activation reconstructor (AR): Take the generated explanation $z$ and produce $h_l \in \mathbb{R}^d$ by running it through the truncated (upto layer $l$) model and using the representation of the last token in $z$. It is like taking a transformer, splliting it into two at layer $l$, where the first part is the activation reconstructor.


## Some initial skepticism
An NLA reads a *single* activation: one token, one layer. But a model's computation is spread across both, so a single $h_l$ is only a local snapshot.

- **Across tokens:** $h_l$ is the running summary up to that token; relevant cognition can sit elsewhere in the sequence. 
- **Across layers:** different information lives at different depths, and the NLA only reads its trained layer. 

## Limitations outlined