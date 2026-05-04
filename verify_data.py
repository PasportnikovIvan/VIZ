import json
import numpy as np

def main():
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("data.json not found. Make sure to run train_and_extract.py first.")
        return
        
    print("--- MNIST CNN Visual Analytics Data Verification ---")
    
    # Classification accuracy for Epoch 0 vs Epoch 15
    correct_epoch_0 = 0
    correct_epoch_15 = 0
    total = len(data)
    
    if total == 0:
        print("data.json is empty.")
        return
    
    # Extract t-SNE points for epoch 15 per class
    tsne_points_e15 = {str(i): [] for i in range(10)}
    
    for item in data:
        true_label = item['true_label']
        
        # Accuracy check
        pred_0 = item['epochs']['0']['predicted_label']
        pred_15 = item['epochs']['15']['predicted_label']
        
        if pred_0 == true_label:
            correct_epoch_0 += 1
        if pred_15 == true_label:
            correct_epoch_15 += 1
            
        # Collect t-SNE points for Epoch 15
        tsne_coords = item['epochs']['15']['tsne']
        tsne_points_e15[str(true_label)].append(tsne_coords)
        
    acc_0 = (correct_epoch_0 / total) * 100
    acc_15 = (correct_epoch_15 / total) * 100
    
    print(f"\nClassification Accuracy ({total} samples):")
    print(f"  Epoch 0 : {acc_0:.2f}%")
    print(f"  Epoch 15: {acc_15:.2f}%")
    
    # Cluster Centroids for t-SNE at Epoch 15
    print("\nt-SNE Centroids (Epoch 15):")
    print(f"{'Class':<6} | {'Avg X':<10} | {'Avg Y':<10}")
    print("-" * 32)
    for c in range(10):
        c_str = str(c)
        pts = np.array(tsne_points_e15[c_str])
        if len(pts) > 0:
            avg_x = np.mean(pts[:, 0])
            avg_y = np.mean(pts[:, 1])
            print(f"{c:<6} | {avg_x:<10.2f} | {avg_y:<10.2f}")
        else:
            print(f"{c:<6} | {'N/A':<10} | {'N/A':<10}")
            
if __name__ == '__main__':
    main()
