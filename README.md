# StepX1Edit — Facade Studio 

An AI-powered system for **instruction-driven architectural facade generation and editing** using diffusion models.  
Developed as the **AMIT Final Project** by **Mai and Aya**.  

---

## Project Overview  
The **StepX1Edit Facade Studio** integrates:  
- **Step1X-Edit** → instruction-based image editing  
- **SDXL-Turbo** → rapid, high-quality image generation  
- **CMP Facade Database Extended (456 images)** → reproducible evaluation  

 Features:  
- Generate **facade concepts** from text prompts.  
- Apply **natural language edits** to existing building facades.  
- Support **dataset-based evaluation** for research & reproducibility.  
- Provide a simple **Gradio UI** with tabs for Generate, Edit, and Dataset Browser.  

⚙️ **Implementation note (compatibility):**  
- Step1X-Edit is the primary instruction editor.  
- If it fails to load (e.g., in Colab), the system falls back to **Instruct-Pix2Pix** (`timbrooks/instruct-pix2pix`).  
- The interface and logs report which model was used.  

---

## Installation  

Clone this repo:  
```bash
git clone https://github.com/yourusername/amit-final-project.git
cd amit-final-project

```
## Install dependencies:
```
pip install -r requirements.txt
```
---
## Run the Notebook

**Open StepX1Edit_Facade_Studio.ipynb in Jupyter or Colab:**
```bash
jupyter notebook Gradio.ipynb
```
---
## Dataset

CMP Facade Database Extended (CMP_facade_DB_extended.zip)

Contains 456 labeled facade images

Used for reproducible evaluation

facade_prompts_200.xlsx

Contains 200 natural language prompts for testing edits

## Example Workflow

1. Load a facade image from the dataset.

2. Enter an instruction, e.g.:
“Add greenery and modernize the facade.”

3. Generate edited results with Step1X-Edit.

---
## Demo

![Demo — StepX1Edit Facade Studio](assets/Demo.gif)
---
## Roadmap & Future Work

 Expand the dataset with more real-world facade images

 Support 3D/BIM integration for architectural workflows

 Improve the semantic consistency of instruction edits

 Deploy as a cloud-hosted service (e.g., Hugging Face Spaces or AWS)
 ---
 
## References

- Hugging Face Diffusers![Demo — StepX1Edit Facade Studio](https://huggingface.co/docs/diffusers)

- SDXL-Turbo

- Step1X-Edit Framework

- CMP Facade Database Extended
---
## Authors

**Mai and Aya**

Final Project — AMIT Institute
