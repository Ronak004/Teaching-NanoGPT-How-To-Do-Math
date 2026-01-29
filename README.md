# Teaching NanoGPT to do Math (DPO Fine-Tuning)

This project fine-tunes a lightweight **NanoGPT** model to solve basic arithmetic and algebra-style equations using **Direct Preference Optimization (DPO)**. The model learns from **positive vs negative answer pairs** to improve its ability to output correct numeric answers for math prompts.

---

## ✨ Key Features

- **Fine-tuned NanoGPT to answer:**
  - Addition: `a+b=?`
  - Subtraction: `a-b=?`
  - Multiplication: `a*b=?`
  - Division (exact integer division): `a/b=?`
  - Linear equations:
    - `x*b=c, x=?`
    - `b-x=c, x=?`
    - `x+b=c, x=?`
- **Preference dataset generation with:**
  - **Correct positive answers** (with explanation format)
  - **Incorrect / weak negative answers** (e.g., “Sorry, I don’t know” or near-miss wrong numbers)
- **Trained using DPO loss** on GPU for fast convergence.
- **Evaluation scripts** included to test model outputs on custom prompts.

---

## ✅ Dataset Format (DPO Training)

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

* **Note:** Only `positive` and `negative` are used during DPO training.
* **Fields** like `Q`, `A`, and `type` are included for evaluation and debugging.

---

## 🧠 Training Method (DPO)

We optimize the model to prefer positive outputs over negative outputs using the DPO loss function:

Where:

*  is the preferred (correct) completion.
*  is the non-preferred (incorrect) completion.
*  is a hyperparameter that controls the preference sharpness.

---

## 🚀 How to Run

1. **Install dependencies:**
```bash
pip install torch numpy tqdm matplotlib

```


2. **Generate dataset (e.g., 100k pairs):**
```bash
python generate_pairs.py

```


3. **Train using DPO:**
Open the notebook `dpo/dpo.ipynb` and run the training cells to produce checkpoints such as:
`dpo/dpo_epoch1.pt` ... `dpo/dpo_epoch10.pt`
4. **Evaluate:**
Run the evaluation cells in the notebook with test prompts:
```python
test_set = ["17+19=?", "3*17=?", "72/4=?", "72-x=34, x=?", "x*11=44, x=?"]

```



---

## 📊 Example Outputs

| Input | Model Output |
| --- | --- |
| **Q:** `72/4=?` | **A:** `18` |
| **Q:** `x*11=44, x=?` | **A:** `4` |

---

## ⚙️ Notes / Debugging

* **Prompt Formatting:** Prompts should match training formatting. Use the cue: `"The answer is "`.
* **Spacing:** In equation prompts, spacing matters. `72-x=34, x=?` may behave differently than `72-x=34,x=?` if the training data is inconsistent.
* **CUDA Errors:** If a device-side assert is triggered, restart the runtime and ensure checkpoints are loaded correctly (try loading on CPU first).

---

## 👨‍💻 Contributors

Completed as part of the **SC3000 (Machine Learning)** assignment. Key contributions include:

* Preference dataset creation (positive/negative pairs).
* DPO training loop implementation.
* Dataset/prompt formatting and debugging.
* Model evaluation and checkpointing.

---

## 📌 Acknowledgements

* **NanoGPT** implementation inspired by [Andrej Karpathy](https://github.com/karpathy/nanoGPT).
* **DPO** methodology based on recent LLM alignment research.

```

```

