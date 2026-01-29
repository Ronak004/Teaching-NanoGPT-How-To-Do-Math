````md
# Teaching NanoGPT to do Math (DPO Fine-Tuning)

This project fine-tunes a lightweight **NanoGPT** model to solve basic arithmetic and algebra-style equations using **Direct Preference Optimization (DPO)**. The model learns from **positive vs negative answer pairs** to improve its ability to output correct numeric answers for math prompts.

---

## Key Features

- Fine-tuned NanoGPT to answer:
  - Addition: `a+b=?`
  - Subtraction: `a-b=?`
  - Multiplication: `a*b=?`
  - Division (exact integer division): `a/b=?`
  - Linear equations:
    - `x*b=c, x=?`
    - `b-x=c, x=?`
    - `x+b=c, x=?`
- Preference dataset generation with:
  - **Correct positive answers** (with explanation format)
  - **Incorrect / weak negative answers** (e.g., “Sorry, I don’t know” or near-miss wrong numbers)
- Trained using **DPO loss** on GPU for fast convergence
- Added evaluation scripts to test model outputs on custom prompts

---

## Dataset Format (DPO Training)

Each training sample is stored as a JSON object:

```json
{
  "positive": "72/4=? The answer is 18 because 72/4 equals 18.",
  "negative": "72/4=? Sorry, I do not know!",
  "Q": "72/4=?",
  "A": "18",
  "type": "a_over_b_q"
}
```

✅ Only `positive` and `negative` are used during DPO training.
📝 Fields like `Q`, `A`, and `type` are included for evaluation/debugging.

---

## Training Method (DPO)

We optimize the model to prefer **positive outputs** over **negative outputs** using the DPO loss:

[
\mathcal{L}*{DPO} = -\log \sigma\left(\frac{\log p*\theta(y^+|x) - \log p_\theta(y^-|x)}{\beta}\right)
]

Where:

* (y^+) is the preferred (correct) completion
* (y^-) is the non-preferred (incorrect) completion
* (\beta) controls preference sharpness

---

## How to Run

### 1) Install dependencies

```bash
pip install torch numpy tqdm matplotlib
```

### 2) Generate dataset (100k pairs)

```bash
python generate_pairs.py
```

### 3) Train using DPO

Open the notebook:

```bash
dpo/dpo.ipynb
```

Run training to produce checkpoints such as:

```text
dpo/dpo_epoch1.pt ... dpo/dpo_epoch10.pt
```

### 4) Evaluate

Example evaluation prompts:

```python
test_set = [
  "17+19=?",
  "3*17=?",
  "72/4=?",
  "72-x=34, x=?",
  "x*11=44, x=?"
]
```

---

## Example Outputs

**Input**

```text
Q: 72/4=?
```

**Model**

```text
A: 18
```

**Input**

```text
Q: x*11=44, x=?
```

**Model**

```text
A: 4
```

---

## Notes / Debugging

* During testing, prompts should match training formatting:

  * Use the cue: `"The answer is "`
* For equation prompts, spacing can matter:

  * `72-x=34, x=?` can behave differently from `72-x=34,x=?` if the training data only contains one format.
* If CUDA errors occur (example: `device-side assert triggered`), restart the runtime and load checkpoints on CPU first before moving to GPU.

---

## 📌 Acknowledgements

* NanoGPT implementation inspired by Andrej Karpathy’s NanoGPT
* DPO based on preference optimization techniques commonly used in LLM alignment research
