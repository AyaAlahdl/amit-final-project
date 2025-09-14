# StepX1Edit — Facade Studio

An AI-powered system for **instruction-driven architectural facade generation and editing** using state-of-the-art diffusion models.  
Developed as the **AMIT Final Project** by **Mai and Aya**.

---

## Project Overview

**StepX1Edit Facade Studio** enables architects, designers, and researchers to:

- Generate new facade concepts from textual prompts.
- Edit existing building facades using natural language instructions.
- Evaluate models reproducibly via a labeled and extended facade dataset.
- Explore results easily through a user-friendly Gradio interface (tabs: Generate, Edit, Dataset Browser).

### Core Components

- **Step1X-Edit** — instruction-based image editing.
- **SDXL-Turbo** — rapid, high-quality image generation.
- **CMP Facade Database Extended (456 images)** — for robust, reproducible evaluation.

**Implementation Note:**  
- Step1X-Edit is the default instruction editor.  
- If it fails to load (e.g., in Colab), the app gracefully falls back to **Instruct-Pix2Pix** (`timbrooks/instruct-pix2pix`).  
- The UI and logs always display which model is active.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/amit-final-project.git
cd amit-final-project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Studio

Open the main notebook in Jupyter or Colab:

```bash
jupyter notebook Gradio.ipynb
```

---

## Dataset

- **CMP Facade Database Extended** (`CMP_facade_DB_extended.zip`):  
  - 456 labeled facade images for evaluation and reproducibility.
- **facade_prompts_200.xlsx**:  
  - 200 natural language prompts for testing instruction-based editing.

---

## Example Workflow

1. Load a facade image from the dataset.
2. Enter an instruction, e.g.:  
   _“Add greenery and modernize the facade.”_
3. Generate edited results using Step1X-Edit (or fallback model).

---

## Demo

![Demo — StepX1Edit Facade Studio](assets/Demo.gif)

---

## Roadmap & Future Work

- Expand the dataset with more real-world facade images.
- Support 3D/BIM integration for architectural workflows.
- Improve the semantic consistency of instruction edits.
- Deploy as a cloud-hosted service (e.g., Hugging Face Spaces or AWS).

---

## References

- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)
- [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit)
- [CMP Facade Database Extended](https://cmp.felk.cvut.cz/~tylecr1/facade/)

---

## Authors

**Mai and Aya**  
_Final Project — AMIT Institute_
