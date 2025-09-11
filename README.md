# StepX1Edit - Facade Studio  

An AI-powered system for **instruction-driven architectural facade generation and editing** using diffusion models.  
Developed as the **AMIT Final Project** by **Mai and Aya**.  

---

## Project Overview  
The **StepX1Edit Facade Studio** integrates:  
- **Step1X-Edit** → for instruction-based image editing  
- **SDXL-Turbo** → for rapid, high-quality image generation  
- **CMP Facade Database Extended (456 images)** → for evaluation  

The system enables **natural language editing of urban facades**, supports **reproducible testing** with datasets, and demonstrates **real-time generation** for architectural and sustainability applications.  

**Implementation note (compatibility)** : We attempted to use Step1X-Edit as the primary instruction editor. Due to intermittent loading/inference issues in some Colab/runtime setups, the notebook includes an automatic fallback to Instruct-Pix2Pix (timbrooks/instruct-pix2pix). The system attempts to load Step1X-Edit first and uses the fallback if loading fails; the UI and logs report which model was used for each run.

---

##  Installation  

1. Clone this repo:  
```bash
git clone https://github.com/yourusername/amit-final-project.git
cd amit-final-project
```
---
## Run the Notebook

**Open StepX1Edit_Facade_Studio.ipynb in Jupyter or Colab:**
```bash
jupyter notebook StepX1Edit_Facade_Studio.ipynb
```
---
## Dataset

CMP Facade Database Extended (CMP_facade_DB_extended.zip)

Contains 456 labeled facade images

Used for reproducible evaluation

facade_prompts_200.xlsx

Contains 200 natural language prompts for testing edits

## Example Workflow

**Load a facade image from the dataset**

Enter an instruction, e.g.: 

“Add greenery and modernize the facade”

Generate edited results with Step1X-Edit

## References

- Hugging Face Diffusers

- SDXL-Turbo

- Step1X-Edit Framework

- CMP Facade Database Extended
---
## Authors

**Mai and Aya**

Final Project — AMIT Institute
