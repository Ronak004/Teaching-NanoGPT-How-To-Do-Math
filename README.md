# Teaching NanoGPT to do Math (DPO Fine-Tuning)

This project fine-tunes a lightweight **NanoGPT** model to solve basic arithmetic and algebra-style equations using **Direct Preference Optimization (DPO)**. The model learns from **positive vs negative answer pairs** to improve its ability to output correct numeric answers for math prompts.

---

## Key Features

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

* **Note:** Only `positive` and `negative` are used during DPO training.
* **Fields** like `Q`, `A`, and `type` are included for evaluation and debugging.

---

## 🧠 Training Method: Direct Preference Optimization (DPO)

Instead of traditional Reinforcement Learning from Human Feedback (RLHF), we optimize the model to prefer correct mathematical logic over incorrect ones using the **DPO Loss Function**. This allows the model to align with "preferred" answers without needing a separate reward model.



### DPO Loss Equation
$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

### Key Variables:
* **$\pi_\theta$**: The policy model we are currently training.
* **$\pi_{ref}$**: The reference model (the base model before DPO).
* **$y_w$ ($y^+$)**: The "winning" or preferred completion (the correct math answer).
* **$y_l$ ($y^-$)**: The "losing" or non-preferred completion (the incorrect/weak answer).
* **$\beta$**: A hyperparameter that controls how much we penalize deviations from the reference model (preference sharpness).
* **$\sigma$**: The sigmoid function.

By maximizing the log-likelihood ratio between the preferred and rejected responses, the model effectively "learns" the boundary between correct mathematical derivation and common errors.

---

## How to Run

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

## Example Outputs

| Input | Model Output |
| --- | --- |
| **Q:** `72/4=?` | **A:** `18` |
| **Q:** `x*11=44, x=?` | **A:** `4` |

---

## Notes / Debugging

* **Prompt Formatting:** Prompts should match training formatting. Use the cue: `"The answer is "`.
* **Spacing:** In equation prompts, spacing matters. `72-x=34, x=?` may behave differently than `72-x=34,x=?` if the training data is inconsistent.
* **CUDA Errors:** If a device-side assert is triggered, restart the runtime and ensure checkpoints are loaded correctly (try loading on CPU first).


---

## Acknowledgements

* **NanoGPT** implementation inspired by [Andrej Karpathy](https://github.com/karpathy/nanoGPT).
* **DPO** methodology based on recent LLM alignment research.


