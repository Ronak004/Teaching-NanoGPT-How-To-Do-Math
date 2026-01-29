Teaching NanoGPT to do Math (DPO Fine-Tuning)

This project fine-tunes a lightweight NanoGPT model to solve basic arithmetic and algebra-style equations using Direct Preference Optimization (DPO). The model learns from positive vs negative answer pairs, improving its ability to output correct numeric answers for math prompts.

✨ Key Features

Fine-tuned NanoGPT to answer:

Addition: a+b=?

Subtraction: a-b=?

Multiplication: a*b=?

Division (exact integer division): a/b=?

Linear equations:

x*b=c, x=?

b-x=c, x=?

x+b=c, x=?

Preference dataset generation with:

Correct positive answers (with explanation format)

Incorrect / weak negative answers (e.g., “Sorry, I don’t know” or near-miss wrong numbers)

Trained using DPO loss on GPU for fast convergence

Added evaluation scripts to test model outputs on custom test prompts


✅ Dataset Format (DPO Training)

Each training sample is stored as a JSON object:

{
  "positive": "72/4=? The answer is 18 because 72/4 equals 18.",
  "negative": "72/4=? Sorry, I do not know!",
  "Q": "72/4=?",
  "A": "18",
  "type": "a_over_b_q"
}


Only positive and negative are used during DPO training.
Fields like Q, A, type are included to support evaluation and analysis.

🧠 Training Method (DPO)

We optimize the model to prefer positive outputs over negative outputs using the DPO loss:

𝐿
𝐷
𝑃
𝑂
=
−
log
⁡
𝜎
(
log
⁡
𝑝
𝜃
(
𝑦
+
∣
𝑥
)
−
log
⁡
𝑝
𝜃
(
𝑦
−
∣
𝑥
)
𝛽
)
L
DPO
	​

=−logσ(
β
logp
θ
	​

(y
+
∣x)−logp
θ
	​

(y
−
∣x)
	​

)

Where:

𝑦
+
y
+
 is the preferred (correct) completion

𝑦
−
y
−
 is the non-preferred (incorrect) completion

𝛽
β controls preference sharpness

🚀 How to Run
1) Install dependencies
pip install torch numpy tqdm matplotlib

2) Generate dataset (100k pairs)
python generate_pairs.py

3) Train using DPO

Open:

dpo/dpo.ipynb


Run the training cells to produce:

dpo/dpo_epoch1.pt ... dpo/dpo_epoch10.pt

4) Evaluate

Example evaluation prompts:

test_set = [
  "17+19=?",
  "3*17=?",
  "72/4=?",
  "72-x=34, x=?",
  "x*11=44, x=?"
]

📊 Example Outputs

Input

Q: 72/4=?


Model

A: 18


Input

Q: x*11=44, x=?


Model

A: 4

⚙️ Notes / Debugging

During testing, prompts must match training formatting to work well:

Use: "The answer is " as a cue

For equation-style prompts, spacing matters (e.g., ", x=?" instead of ",x=?")

If CUDA errors occur (device-side assert triggered), restart runtime and load checkpoints on CPU first before moving to GPU.


📌 Acknowledgements

NanoGPT implementation inspired by Andrej Karpathy’s NanoGPT

DPO concept based on preference optimization methods used in alignment research
