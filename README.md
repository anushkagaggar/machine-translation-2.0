# Neural Machine Translation using Transformer (End-to-End Deployment)

## Problem Statement

The goal of this project is to **design, train, and deploy a Transformer-based Neural Machine Translation (NMT) system** from scratch, focusing on understanding the **complete engineering and modeling flow** rather than optimizing translation quality.

This project demonstrates how a modern Transformer-based sequence-to-sequence model is:
- Tokenized using SentencePiece
- Trained with custom encoder–decoder architecture
- Served through a production-ready API
- Exposed via a simple frontend
- Containerized and deployed on a cloud VM (AWS EC2)

The primary objective is **learning the full lifecycle of a Transformer project**, from model training to real-world deployment.

---

## Key Objectives

- Build a Transformer encoder–decoder model from scratch using PyTorch
- Understand attention masks, padding masks, and decoding flow
- Serve the model using FastAPI
- Create a lightweight frontend using Streamlit
- Containerize the application with Docker
- Deploy and run the system on an AWS EC2 instance

---

## Development Phases

### Phase 1: Dataset Preparation & Tokenization
- Parallel source–target text dataset
- SentencePiece tokenizer (subword-based)
- Special tokens: BOS, EOS, PAD
- Fixed maximum sequence length handling

---

### Phase 2: Model Architecture
- Custom Transformer implementation using PyTorch
- Encoder–decoder architecture with:
  - Multi-head self-attention
  - Cross-attention
  - Positional encoding
- Proper handling of:
  - Padding masks
  - Causal (future) masks
- Greedy decoding for inference

---

### Phase 3: Training & Evaluation
- Teacher forcing during training
- Cross-entropy loss with PAD masking
- Checkpointing best model
- BLEU score evaluation (for pipeline completeness)

> Note: Translation quality is **not the focus**; correctness of the pipeline is.

---

### Phase 4: Inference Pipeline
- SentencePiece tokenization at inference time
- Greedy decoding loop
- Proper handling of:
  - EOS stopping condition
  - Device placement (CPU/GPU-safe)

---

### Phase 5: API Development (FastAPI)
- REST API for translation
- `/translate` endpoint for inference
- `/docs` Swagger UI for testing
- Model loaded once at startup for efficiency

---

### Phase 6: Frontend (Streamlit)
- Simple UI to input text
- Sends requests to FastAPI backend
- Displays translated output
- Designed for quick manual testing and demos

---

### Phase 7: Docker Containerization
- Single Docker image running:
  - FastAPI backend
  - Streamlit frontend
- Supervisor used to manage multiple processes
- CPU-only PyTorch build for portability
- Local container testing before cloud deployment

---

### Phase 8: AWS EC2 Deployment
- Deployed on **AWS EC2 (Ubuntu LTS)**
- Docker installed on EC2 instance
- Container pulled from Amazon ECR
- Ports exposed via EC2 Security Group:
  - `8000` → FastAPI
  - `8501` → Streamlit
- Verified access using public IPv4 address

---

## Live Deployment URLs (EC2)

> ⚠️ These URLs depend on the EC2 instance being **running**

### FastAPI (Backend)
http://65.0.21.181:8000/docs


### Streamlit (Frontend)
http://65.0.21.181:8501/


---

## What This Project Demonstrates

- Practical understanding of Transformer internals
- Real-world issues in inference, masking, and decoding
- API + frontend integration for ML models
- Docker-based deployment strategy
- End-to-end ML system engineering mindset

---

## Disclaimer

This project prioritizes **learning, architecture clarity, and deployment flow** over translation accuracy.  
The predictions may not be linguistically meaningful, and that is **intentional** for this phase.

---

## Author

**Anushka Gaggar**  
Machine Learning & NLP Practitioner  
Focused on building end-to-end, production-oriented ML systems
