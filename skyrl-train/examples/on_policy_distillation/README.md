# On-Policy Distillation
This folder contains scripts for running On-Policy Distillation on SkyRL.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6ecc9da-cb67-4935-b95a-193ba6c4843c" width="45%" />
  <img src="https://github.com/user-attachments/assets/255c46bd-f443-4f36-b8fc-9546c3a01cf1" width="45%" />
</p>

On-Policy distillation has been shown to be an effective technique for efficiently post-training models ([Agarwal et al](https://arxiv.org/abs/2306.13649), [Gu et al](https://arxiv.org/abs/2306.08543), [Qwen3 team](https://arxiv.org/abs/2505.09388), [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation)).

`main_on_policy_distill.py` provides a simple example for modifying SkyRL to implement On-Policy distillation by replacing the ref model with a teacher model, and making some minor modifications to the reward/loss logic!
