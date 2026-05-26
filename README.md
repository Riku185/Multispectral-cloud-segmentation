
Cloud Mask Project (Complete Version)

Workflow:

1. Put ALL original TIFF files (TOA + CLD) into:
   data/raw/all/

2. Run dataset preparation:
   python prepare_dataset.py

3. Install requirements:
   pip install -r requirements.txt

4. Start training:
   python main.py --stage train

Output:
best_cloud_model.pth
