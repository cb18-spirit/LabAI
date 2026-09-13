# 🔬 LabAI – AI-Powered Electronics Laboratory Assistant

<div align="center">

### Intelligent Laboratory Learning, Experiment Diagnostics & Research Assistant for Electronics and Communication Engineering

Built using **Streamlit**, **Gemini AI**, **FAISS**, **Sentence Transformers**, and **MongoDB Atlas**

</div>

---

# Overview

LabAI is an AI-powered educational platform designed specifically for Electronics and Communication Engineering laboratories.

Instead of providing generic chatbot responses, LabAI combines:

* Laboratory Manuals
* Experiment Databases
* Fault Diagnosis Knowledge Bases
* Research Paper Collections
* AI Reasoning

to provide accurate, context-aware assistance for students and laboratory technicians.

---

# Key Features

## 🎓 Student Learning Assistant

Students can:

* Ask questions about experiments
* Understand laboratory procedures
* Learn experiment theory
* Prepare for viva examinations
* Clarify observations and results

Example:

> Explain the working of a Band Pass Filter.

---

## 🔍 Experiment Troubleshooting

Students can upload:

* Circuit Images
* Input Voltage
* Output Voltage
* Frequency
* Observed Results

LabAI analyzes the data and suggests likely faults.

Example:

```text
Observed Output: 1.2V
Expected Output: 5V

Possible Causes:
✓ Wrong resistor value
✓ Faulty capacitor
✓ Incorrect wiring
✓ Grounding issue
```

---

## 📷 Circuit Image Analysis

LabAI supports:

* Circuit photo upload
* Edge detection analysis
* Circuit comparison
* Visual troubleshooting support

Using:

* OpenCV
* Computer Vision

---

## 📚 Communication Systems RAG Assistant

LabAI contains a Retrieval-Augmented Generation (RAG) database built from Communication Systems laboratory manuals.

Students can ask:

* ASK experiments
* PSK experiments
* FSK experiments
* PCM experiments
* Delta Modulation experiments
* Communication system theory

and receive answers grounded in actual laboratory content.

---

## ⚡ Analog Electronic Circuits (AEC) Assistant

Dedicated AEC knowledge base includes:

* Regulated Power Supply
* Class B Push Pull Amplifier
* DAC Experiments
* Op-Amp Circuits
* Analog Design Concepts

Students receive answers directly from the AEC laboratory manual instead of generic internet responses.

---

## 📄 Research Paper Assistant

LabAI includes a research paper knowledge base built using:

* FAISS
* Sentence Transformers
* Gemini AI

Current research areas include:

* OFDM
* OFDM Index Modulation
* Massive MIMO
* Cell-Free Massive MIMO
* Full Duplex Communication
* OTFS
* PAPR Reduction
* Multicarrier Communications

Capabilities:

* Semantic paper search
* Research summaries
* Literature review assistance
* Concept explanations
* Source-based answers

Example:

```text
What is OFDM Index Modulation?
```

LabAI retrieves relevant research chunks before generating a response.

---

## 🧠 Research Chat

Students can:

* Search research papers
* View retrieved paper content
* Ask follow-up questions
* Generate research summaries

Example:

```text
Compare OFDM and OTFS
```

---

## 👨‍🔧 Lab Technician Dashboard

Lab technicians can:

* Review experiment submissions
* Approve or reject content
* Manage laboratory knowledge
* Expand experiment repositories

---

## 📝 Student Experiment Submission

Students can contribute:

* New experiments
* Experiment notes
* Laboratory guides
* PDFs and documentation

Workflow:

```text
Student Submission
        ↓
Pending Review
        ↓
Lab Technician Approval
        ↓
Published Repository
```

---

# System Architecture

```text
                     ┌───────────────┐
                     │    Student    │
                     └───────┬───────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      LabAI      │
                    │  Streamlit UI   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼

 ┌────────────┐      ┌───────────────┐     ┌──────────────┐
 │ MongoDB    │      │ FAISS Vector  │     │ Gemini AI    │
 │ Atlas      │      │ Databases     │     │ Reasoning    │
 └────────────┘      └───────────────┘     └──────────────┘
```

---

# Technology Stack

| Component          | Technology            |
| ------------------ | --------------------- |
| Frontend           | Streamlit             |
| Authentication     | MongoDB Atlas         |
| Database           | MongoDB               |
| AI Model           | Gemini                |
| Embeddings         | Sentence Transformers |
| Vector Search      | FAISS                 |
| Image Processing   | OpenCV                |
| Research Retrieval | RAG Pipeline          |
| Backend            | Python                |

---

# Project Structure

```text
LabAI/
│
├── app.py
├── auth.py
├── database.py
│
├── experiments/
│   ├── combined_database/
│   ├── experiment_1/
│   ├── experiment_2/
│   ├── experiment_3/
│   ├── experiment_4/
│   ├── experiment_5/
│   ├── experiment_6/
│   └── experiment_7/
│
├── aec/
│   ├── AEC.pdf
│   ├── aec_chunks.json
│   ├── aec_embeddings.npy
│   └── aec_faiss.index
│
├── research/
│   ├── research_chunks.json
│   ├── research_embeddings.npy
│   └── research_faiss.index
│
├── reference_circuits/
│
├── knowledge_base.json
├── fault_dataset.json
│
└── requirements.txt
```

---

# Security

User passwords are securely stored using:

* bcrypt password hashing
* MongoDB Atlas authentication
* Session-based login management

No plaintext passwords are stored.

---

# Future Roadmap

### Phase 1

* Multi-lab support
* Advanced fault diagnosis
* Better circuit image analysis

### Phase 2

* Research paper recommendation engine
* Citation generation
* Literature review generation

### Phase 3

* AI-generated viva questions
* AI-generated experiment reports
* Personalized learning paths

### Phase 4

* FPGA Laboratory Assistant
* VLSI Laboratory Assistant
* Embedded Systems Laboratory Assistant
* IoT Laboratory Assistant

---

# Impact

LabAI helps students:

* Learn faster
* Troubleshoot independently
* Access research knowledge
* Prepare for vivas
* Understand experiments deeply

while helping laboratory staff manage and scale educational resources efficiently.

---

# Team
* Chethan B
* Dhanush R
* Dhanyashri DK
* Harsha SM

**LabAI**

AI-Powered Electronics Laboratory Learning Platform

Built for enhancing laboratory education through Retrieval-Augmented Generation, Research Intelligence, and Experiment Diagnostics.

---

### ⭐ If you found this project useful, consider starring the repository.
