# Workflow-CI: MLflow Project dengan GitHub Actions

Repository untuk automated model retraining menggunakan MLflow Projects dan GitHub Actions CI/CD.

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-training.yml          # GitHub Actions workflow
├── MLProject/
│   ├── modelling.py                     # Script training model
│   ├── conda.yaml                       # Conda environment
│   ├── MLProject                        # MLflow project config
│   ├── Dockerfile                       # Docker container config
│   ├── indo_spam_preprocessing.csv      # Dataset preprocessing
│   └── vectorizer.joblib                # TF-IDF vectorizer
├── README.md                            # Dokumentasi ini
└── Docker_Hub_Link.txt                  # Link ke Docker Hub

```

## Cara Menggunakan

### 1. Local Testing

Jalankan MLflow project secara lokal:

```bash
cd Workflow-CI
mlflow run MLProject
```

Dengan parameter custom:

```bash
mlflow run MLProject -P test_size=0.3 -P alpha=0.5
```

### 2. GitHub Actions CI/CD

Workflow akan otomatis berjalan ketika:
- Push ke branch `main`
- Pull request ke branch `main`
- Manual trigger via GitHub UI

Workflow akan:
1. Setup environment Python dan Conda
2. Train model menggunakan MLflow
3. Upload model artifacts ke GitHub
4. Build Docker image
5. Push image ke Docker Hub

### 3. Setup GitHub Secrets

Untuk menggunakan Docker Hub integration, tambahkan secrets di GitHub:

1. Go to: Repository → Settings → Secrets and variables → Actions
2. Tambahkan:
   - `DOCKERHUB_USERNAME` - Username Docker Hub Anda
   - `DOCKERHUB_TOKEN` - Access token dari Docker Hub

### 4. Docker Usage

Pull image from Docker Hub:

```bash
docker pull YOUR_DOCKERHUB_USERNAME/workflow-ci-spam-detection:latest
```

Run container:

```bash
docker run -p 5000:5000 YOUR_DOCKERHUB_USERNAME/workflow-ci-spam-detection:latest
```

## Model Details

- **Model**: Multinomial Naive Bayes
- **Dataset**: Indonesian Spam Detection
- **Features**: TF-IDF Vectorization
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

