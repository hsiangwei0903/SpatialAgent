# SpatialAgent

**1st Place Solution of the ICCV 2025 AI CIty Challenge, Track 3.**

<p align="center">
  <img src="asset/leaderboard.png" alt="Leaderboard Result"/>
</p>

---

## 🔧 Installation

1. Clone the repository  

       git clone https://github.com/hsiangwei0903/SpatialAgent.git
       cd SpatialAgent

2. Create and activate a conda environment with Python 3.10

       conda create -n spatialagent python=3.10 -y
       conda activate spatialagent

3. Install Python dependencies (Adjust pytorch installation with your CUDA version)

       pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
       pip install -r requirements.txt

5. Install Google API and Vertex AI packages following the [official guide](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#gen-ai-sdk-for-python).

---

## 📦 Preparation

1. Model checkpoints and pre-processed QA data can be downloaded from [here](<https://drive.google.com/drive/u/1/folders/1_ovPjqADpvM0fQdNBLAPdWiemC5MFaG7>).

2. Place the downloaded files in corresponding directory following the below Project Structure.

3. Setup a [Vertex AI API project ID](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#gen-ai-sdk-for-python).

4. Download the [AI City Challenge PhysicalAI Spatial Intelligence dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse) and put in data dir following project structure.

---

## 📂 Project Structure

    SpatialAgent
    ├── agent
    ├── distance_est/
    │   └──  ckpt/
    │       ├── 3m_epoch6.pth
    │       └── epoch_5_iter_6831.pth
    ├── inside_pred/
    │   └── ckpt/
    │       └── epoch_4.pth
    ├── utils
    ├── data/
    │   ├── train
    │   ├── val
    │   └── test/
    │       └── images/
    │       └── depths/
    └── README.md

---

## 🧠 Usage

### 1. Inference on test set (For full reproduce of our results)

```

cd agent
python agent_run.py --project_id <your Vertex AI API>

```
Additionally, some QA might failed because Gemini return invalid format or answer, run again with thinking mode enabled can solve this issue. 
Running this command will re-run those failure cases.
```

cd agent
python agent_run.py --project_id <your Vertex AI API> --think_mode

```

## ⚒️ QA Data Pre-processing and Model Training (Optional)

### 0. QA Data Pre-processing

To pre-process the QA, you need to update the below script with your Google API key.
Note that this step is optional because data.zip already provide the processed QA data.

```
python utils/question_rephrase.py
```


We provide the pre-trained model checkpoint, but we also provide the training script of our model as follows.

### 1. Train the distance estimation model

```
cd distance_est
python train.py
```

### 2. Train the inclusion classification model

```
cd inside_pred
python train.py
```
