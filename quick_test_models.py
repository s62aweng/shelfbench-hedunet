"""
quick check to see how models have done
"""

from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import cv2
from data_processing.ice_data import IceDataset


model_path = '/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs/FPN_model_epoch_49.pth'
parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/preprocessed_data/"  # Update with your data path
output_dir = '/home/users/amorgan/benchmark_CB_AM/figures/'

CLASS_NAMES = ["Other", "Land ice"]
N_CLASSES = len(CLASS_NAMES)
from matplotlib.colors import ListedColormap
COLORMAP = ListedColormap(['blue', 'lightgray'])

# Load  model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_prediction(image, mask, pred, class_names=None, save_path=None, sample_idx=0):
    """
    Visualize the original image, ground truth mask, prediction, and metrics
    Args:
        image: normalized tensor image
        mask: ground truth mask tensor
        pred: prediction mask tensor
        class_names: list of class names for legend
        save_path: path to save the visualization
        sample_idx: index of the sample being visualized
    """
    print(f"Sample {sample_idx} - Image shape: {image.shape}")
    print(f"Sample {sample_idx} - Mask shape: {mask.shape}")
    print(f"Sample {sample_idx} - Prediction shape: {pred.shape}")
  
    # Convert tensors to numpy arrays
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    print(f"Sample {sample_idx} - After squeeze - Image: {image.shape}, Mask: {mask.shape}, Pred: {pred.shape}")
    
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    
    # Calculate comprehensive metrics
    pixel_accuracy = (pred == mask).mean()
    
    # Calculate per-class IoU and metrics
    class_ious, class_accuracies = calculate_class_metrics(pred, mask)
    mean_iou = np.mean(class_ious)
    
    # Calculate additional metrics
    dice_scores = calculate_dice_scores(pred, mask)
    precision_scores = calculate_precision(pred, mask)
    recall_scores = calculate_recall(pred, mask)
    
    # Set up matplotlib for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False  # Set to True if you have LaTeX installed
    })
    
    # Create figure with 4 columns and proper spacing
    fig = plt.figure(figsize=(20, 8))
    
    # Create a grid with custom spacing
    gs = fig.add_gridspec(2, 4, height_ratios=[4, 1], width_ratios=[1, 1, 1, 1.2], 
                         hspace=0.3, wspace=0.3)
    
    # Main plots in top row
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Plot original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('(a) Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot ground truth mask
    im1 = ax2.imshow(mask, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
    ax2.set_title('(b) Ground Truth', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Plot prediction
    im2 = ax3.imshow(pred, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
    ax3.set_title('(c) Prediction', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Metrics visualization in 4th column
    ax4.axis('off')
    ax4.set_title('(d) Quantitative Metrics', fontsize=14, fontweight='bold')
    
    # Create metrics text with proper formatting
    metrics_text = f"""
Overall Metrics:
• Pixel Accuracy: {pixel_accuracy:.3f}
• Mean IoU: {mean_iou:.3f}

Per-Class IoU:
• {class_names[0]}: {class_ious[0]:.3f}
• {class_names[1]}: {class_ious[1]:.3f}

Per-Class Dice Score:
• {class_names[0]}: {dice_scores[0]:.3f}
• {class_names[1]}: {dice_scores[1]:.3f}

Per-Class Precision:
• {class_names[0]}: {precision_scores[0]:.3f}
• {class_names[1]}: {precision_scores[1]:.3f}

Per-Class Recall:
• {class_names[0]}: {recall_scores[0]:.3f}
• {class_names[1]}: {recall_scores[1]:.3f}
    """.strip()
    
    # Add metrics text to the 4th subplot
    ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Create a separate colorbar axis in the bottom row
    cbar_ax = fig.add_subplot(gs[1, :3])  # Spans first 3 columns of bottom row
    cbar_ax.axis('off')
    
    # Add colorbar with better positioning
    cbar = plt.colorbar(im2, ax=cbar_ax, orientation='horizontal', 
                       fraction=0.8, pad=0.1, aspect=30)
    cbar.set_label('Class Labels', fontsize=12, fontweight='bold')
    
    # Customize colorbar ticks
    tick_positions = np.arange(N_CLASSES)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{i}: {name}' for i, name in enumerate(class_names)])
    
    # Add sample information as figure title
    fig.suptitle(f'Ice Segmentation Results - Sample {sample_idx}', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(top=0.88, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Saved publication-quality visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)

def calculate_class_metrics(pred, mask, num_classes=2):
    """Calculate IoU and accuracy for each class"""
    ious = []
    accuracies = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        mask_cls = (mask == cls)
        
        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        ious.append(iou)
        
        # Class-specific accuracy
        if mask_cls.sum() > 0:
            acc = (pred_cls & mask_cls).sum() / mask_cls.sum()
        else:
            acc = 1.0 if pred_cls.sum() == 0 else 0.0
        
        accuracies.append(acc)
    
    return ious, accuracies

def calculate_dice_scores(pred, mask, num_classes=2):
    """Calculate Dice coefficient for each class"""
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        mask_cls = (mask == cls)
        
        intersection = (pred_cls & mask_cls).sum()
        total = pred_cls.sum() + mask_cls.sum()
        
        if total > 0:
            dice = (2.0 * intersection) / total
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        dice_scores.append(dice)
    
    return dice_scores

def calculate_precision(pred, mask, num_classes=2):
    """Calculate precision for each class"""
    precisions = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        mask_cls = (mask == cls)
        
        true_positive = (pred_cls & mask_cls).sum()
        false_positive = (pred_cls & ~mask_cls).sum()
        
        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 1.0 if true_positive == 0 else 0.0
        
        precisions.append(precision)
    
    return precisions

def calculate_recall(pred, mask, num_classes=2):
    """Calculate recall (sensitivity) for each class"""
    recalls = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        mask_cls = (mask == cls)
        
        true_positive = (pred_cls & mask_cls).sum()
        false_negative = (~pred_cls & mask_cls).sum()
        
        if (true_positive + false_negative) > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 1.0 if true_positive == 0 else 0.0
        
        recalls.append(recall)
    
    return recalls

def create_comprehensive_metrics_plot(all_ious, all_pixel_accuracies, all_class_ious, output_dir):
    """Create a comprehensive metrics visualization with publication quality"""
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    
    avg_iou = np.mean(all_ious)
    avg_pixel_accuracy = np.mean(all_pixel_accuracies)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # IoU distribution
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(all_ious, bins=15, alpha=0.7, color='skyblue', 
                               edgecolor='black', linewidth=1.2)
    ax1.axvline(avg_iou, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean IoU: {avg_iou:.4f}')
    ax1.set_xlabel('IoU Score', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('(a) IoU Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Pixel accuracy distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_pixel_accuracies, bins=15, alpha=0.7, color='lightgreen', 
            edgecolor='black', linewidth=1.2)
    ax2.axvline(avg_pixel_accuracy, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean Accuracy: {avg_pixel_accuracy:.4f}')
    ax2.set_xlabel('Pixel Accuracy', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('(b) Pixel Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Per-class IoU comparison
    ax3 = fig.add_subplot(gs[1, 0])
    class_mean_ious = [np.mean(all_class_ious[i]) if all_class_ious[i] else 0 
                      for i in range(N_CLASSES)]
    colors = ['coral', 'lightblue']
    bars = ax3.bar(CLASS_NAMES, class_mean_ious, alpha=0.8, 
                  color=colors[:N_CLASSES], edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Mean IoU', fontweight='bold')
    ax3.set_title('(c) Per-Class IoU Performance', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=0)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, value in zip(bars, class_mean_ious):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # IoU vs Pixel Accuracy scatter
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(all_ious, all_pixel_accuracies, alpha=0.7, 
                         color='purple', s=50, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('IoU Score', fontweight='bold')
    ax4.set_ylabel('Pixel Accuracy', fontweight='bold')
    ax4.set_title('(d) IoU vs Pixel Accuracy Correlation', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add correlation coefficient
    correlation = np.corrcoef(all_ious, all_pixel_accuracies)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax4.transAxes, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['IoU Score', f'{np.mean(all_ious):.4f}', f'{np.std(all_ious):.4f}', 
         f'{np.min(all_ious):.4f}', f'{np.max(all_ious):.4f}'],
        ['Pixel Accuracy', f'{np.mean(all_pixel_accuracies):.4f}', 
         f'{np.std(all_pixel_accuracies):.4f}', f'{np.min(all_pixel_accuracies):.4f}', 
         f'{np.max(all_pixel_accuracies):.4f}']
    ]
    
    for i, class_name in enumerate(CLASS_NAMES):
        if all_class_ious[i]:
            class_ious_list = all_class_ious[i]
            table_data.append([f'{class_name} IoU', f'{np.mean(class_ious_list):.4f}',
                             f'{np.std(class_ious_list):.4f}', f'{np.min(class_ious_list):.4f}',
                             f'{np.max(class_ious_list):.4f}'])
    
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0.1, 0.3, 0.8, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('(e) Summary Statistics', fontsize=14, fontweight='bold', y=0.85)
    
    plt.suptitle('Comprehensive Model Performance Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(output_dir, 'comprehensive_metrics_publication.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Publication-quality metrics plot saved to {os.path.join(output_dir, 'comprehensive_metrics_publication.png')}")
    plt.close()
    
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model - you may need to adjust architecture parameters
    try:
        model = smp.FPN(
            encoder_name="resnet50", 
            encoder_weights="imagenet",
            in_channels=1,
            classes=2,
        )
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from checkpoint with 'model_state_dict' key")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model from checkpoint with 'state_dict' key")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
            
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please verify your model architecture matches the saved checkpoint")
        return
    
    # Create test dataset
    try:
        val_dataset = IceDataset(mode='val', parent_dir=parent_dir)
        print(f"val dataset loaded with {len(val_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Generate predictions for samples
    num_samples = min(10, len(val_dataset))  # Visualize up to 10 samples
    print(f"Generating predictions for {num_samples} test samples...")
    
    # Store metrics for analysis
    all_ious = []
    all_pixel_accuracies = []
    all_class_ious = {i: [] for i in range(N_CLASSES)}
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                # Get a sample
                image, mask = val_dataset[i]
                
                # Add batch dimension and move to device
                image_batch = image.unsqueeze(0).to(device)
                mask = mask.to(device)
                
                # Get prediction
                output = model(image_batch)
                
                # Convert output probabilities to class predictions
                pred = torch.argmax(output, dim=1).squeeze(0)
                # FIX: Flip the predicted classes (0 becomes 1, 1 becomes 0)
                pred = 1 - pred  # This flips 0->1 and 1->0

                # Move back to CPU for visualization
                pred_np = pred.cpu().numpy()
                mask_np = mask.cpu().numpy()
                
                print(f"Sample {i} - Unique classes in mask: {np.unique(mask_np)}")
                print(f"Sample {i} - Unique classes in pred: {np.unique(pred_np)}")
                
                # Calculate metrics
                pixel_accuracy = (pred_np == mask_np).mean()
                all_pixel_accuracies.append(pixel_accuracy)
                
                # Calculate IoU per class
                class_ious, class_accs = calculate_class_metrics(pred_np, mask_np)
                for cls, iou in enumerate(class_ious):
                    all_class_ious[cls].append(iou)
                
                mean_iou = np.mean(class_ious)
                all_ious.append(mean_iou)
                
                print(f"Sample {i} - IoU: {mean_iou:.4f}, Pixel Acc: {pixel_accuracy:.4f}")
                
                # Visualize every sample with enhanced visualization
                save_path = os.path.join(output_dir, f'prediction_sample_{i}_enhanced.png')
                visualize_prediction(image, mask, pred, CLASS_NAMES, save_path, i)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    print(f"saved to {output_dir}")
    
    # Print comprehensive metrics
    if all_ious:
        avg_iou = np.mean(all_ious)
        avg_pixel_accuracy = np.mean(all_pixel_accuracies)
        
        print(f"\n=== Overall Metrics ===")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")
        
        print(f"\n=== Per-Class IoU ===")
        for cls, class_name in enumerate(CLASS_NAMES):
            if all_class_ious[cls]:
                cls_iou = np.mean(all_class_ious[cls])
                print(f"{class_name}: {cls_iou:.4f}")
        
        # Create enhanced comprehensive metrics plot
        create_comprehensive_metrics_plot(all_ious, all_pixel_accuracies, 
                                        all_class_ious, output_dir)

if __name__ == "__main__":
    main()



# simple old version

# from PIL import Image
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import segmentation_models_pytorch as smp
# from torchvision import transforms
# import cv2
# from data_processing.ice_data import IceDataset


# model_path = '/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs/FPN_model_epoch_49.pth'
# parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/preprocessed_data/"  # Update with your data path
# output_dir = '/home/users/amorgan/benchmark_CB_AM/figures/'

# CLASS_NAMES = ["Other", "Land ice"]
# N_CLASSES = len(CLASS_NAMES)
# from matplotlib.colors import ListedColormap
# COLORMAP = ListedColormap(['blue', 'lightgray'])

# # Load  model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def visualize_prediction(image, mask, pred, class_names=None, save_path=None, sample_idx=0):
#     """
#     Visualize the original image, ground truth mask, and prediction
#     Args:
#         image: normalized tensor image
#         mask: ground truth mask tensor
#         pred: prediction mask tensor
#         class_names: list of class names for legend
#         save_path: path to save the visualization
#         sample_idx: index of the sample being visualized
#     """
#     print(f"Sample {sample_idx} - Image shape: {image.shape}")
#     print(f"Sample {sample_idx} - Mask shape: {mask.shape}")
#     print(f"Sample {sample_idx} - Prediction shape: {pred.shape}")
  
#     # Convert tensors to numpy arrays
#     image = image.squeeze().cpu().numpy()
#     mask = mask.squeeze().cpu().numpy()
#     pred = pred.squeeze().cpu().numpy()

#     print(f"Sample {sample_idx} - After squeeze - Image: {image.shape}, Mask: {mask.shape}, Pred: {pred.shape}")
    
#     # Clip to [0, 1] range
#     image = np.clip(image, 0, 1)
    
#     # Create figure with better spacing
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Plot original image
#     axes[0].imshow(image, cmap='gray')
#     axes[0].set_title(f'Original Image (Sample {sample_idx})', fontsize=14)
#     axes[0].axis('off')
    
#     # Plot ground truth mask
#     im1 = axes[1].imshow(mask, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
#     axes[1].set_title('Ground Truth Mask', fontsize=14)
#     axes[1].axis('off')
    
#     # Plot prediction
#     im2 = axes[2].imshow(pred, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
#     axes[2].set_title('Prediction', fontsize=14)
#     axes[2].axis('off')

#     # Add colorbar
#     plt.colorbar(im2, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    
#     # Add legend with class names
#     if class_names:
#         import matplotlib.patches as mpatches
#         patches = []
#         for i, name in enumerate(class_names):
#             color = COLORMAP(i / (N_CLASSES-1))
#             patches.append(mpatches.Patch(color=color, label=f'{i}: {name}'))
#         fig.legend(handles=patches, loc='lower center', ncol=N_CLASSES, 
#                   bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f"Saved visualization to {save_path}")
#     else:
#         plt.show()
    
#     plt.close(fig)

# def calculate_class_metrics(pred, mask, num_classes=2):
#     """Calculate IoU and accuracy for each class"""
#     ious = []
#     accuracies = []
    
#     for cls in range(num_classes):
#         pred_cls = (pred == cls)
#         mask_cls = (mask == cls)
        
#         intersection = (pred_cls & mask_cls).sum()
#         union = (pred_cls | mask_cls).sum()
        
#         if union > 0:
#             iou = intersection / union
#         else:
#             iou = 1.0 if intersection == 0 else 0.0
        
#         ious.append(iou)
        
#         # Class-specific accuracy
#         if mask_cls.sum() > 0:
#             acc = (pred_cls & mask_cls).sum() / mask_cls.sum()
#         else:
#             acc = 1.0 if pred_cls.sum() == 0 else 0.0
        
#         accuracies.append(acc)
    
#     return ious, accuracies

# def main():
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Load the model - you may need to adjust architecture parameters
#     try:
#         model = smp.FPN(
#             encoder_name="resnet50", 
#             encoder_weights="imagenet",
#             in_channels=1,
#             classes=2,
#         )
        
#         # Load the trained weights
#         checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
#         # Handle different checkpoint formats
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             print("Loaded model from checkpoint with 'model_state_dict' key")
#         elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['state_dict'])
#             print("Loaded model from checkpoint with 'state_dict' key")
#         else:
#             model.load_state_dict(checkpoint)
#             print("Loaded model state dict directly")
            
#         model.to(device)
#         model.eval()
#         print("Model loaded successfully!")
        
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Please verify your model architecture matches the saved checkpoint")
#         return
    
#     # Create test dataset
#     try:
#         val_dataset = IceDataset(mode='val', parent_dir=parent_dir)
#         print(f"Test dataset loaded with {len(val_dataset)} samples")
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return
    
#     # Generate predictions for samples
#     num_samples = min(10, len(val_dataset))  # Visualize up to 10 samples
#     print(f"Generating predictions for {num_samples} test samples...")
    
#     # Store metrics for analysis
#     all_ious = []
#     all_pixel_accuracies = []
#     all_class_ious = {i: [] for i in range(N_CLASSES)}
    
#     with torch.no_grad():
#         for i in range(num_samples):
#             try:
#                 # Get a sample
#                 image, mask = val_dataset[i]
                
#                 # Add batch dimension and move to device
#                 image_batch = image.unsqueeze(0).to(device)
#                 mask = mask.to(device)
                
#                 # Get prediction
#                 output = model(image_batch)
                
#                 # Convert output probabilities to class predictions
#                 pred = torch.argmax(output, dim=1).squeeze(0)
#                 # FIX: Flip the predicted classes (0 becomes 1, 1 becomes 0)
#                 pred = 1 - pred  # This flips 0->1 and 1->0


#                 # Move back to CPU for visualization
#                 pred_np = pred.cpu().numpy()
#                 mask_np = mask.cpu().numpy()
                
#                 print(f"Sample {i} - Unique classes in mask: {np.unique(mask_np)}")
#                 print(f"Sample {i} - Unique classes in pred: {np.unique(pred_np)}")
                
#                 # Calculate metrics
#                 pixel_accuracy = (pred_np == mask_np).mean()
#                 all_pixel_accuracies.append(pixel_accuracy)
                
#                 # Calculate IoU per class
#                 class_ious, class_accs = calculate_class_metrics(pred_np, mask_np)
#                 for cls, iou in enumerate(class_ious):
#                     all_class_ious[cls].append(iou)
                
#                 mean_iou = np.mean(class_ious)
#                 all_ious.append(mean_iou)
                
#                 print(f"Sample {i} - IoU: {mean_iou:.4f}, Pixel Acc: {pixel_accuracy:.4f}")
                
#                 # Visualize every sample (or adjust frequency as needed)
#                 save_path = os.path.join(output_dir, f'prediction_sample_{i}.png')
#                 visualize_prediction(image, mask, pred, CLASS_NAMES, save_path, i)
                
#             except Exception as e:
#                 print(f"Error processing sample {i}: {e}")
#                 continue
    
#     print(f"Visualizations saved to {output_dir}")
    
#     # Print comprehensive metrics
#     if all_ious:
#         avg_iou = np.mean(all_ious)
#         avg_pixel_accuracy = np.mean(all_pixel_accuracies)
        
#         print(f"\n=== Overall Metrics ===")
#         print(f"Average IoU: {avg_iou:.4f}")
#         print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")
        
#         print(f"\n=== Per-Class IoU ===")
#         for cls, class_name in enumerate(CLASS_NAMES):
#             if all_class_ious[cls]:
#                 cls_iou = np.mean(all_class_ious[cls])
#                 print(f"{class_name}: {cls_iou:.4f}")
        
#         # Plot metrics distribution
#         plt.figure(figsize=(15, 10))
        
#         # IoU distribution
#         plt.subplot(2, 2, 1)
#         plt.hist(all_ious, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
#         plt.axvline(avg_iou, color='r', linestyle='--', linewidth=2, 
#                    label=f'Mean IoU: {avg_iou:.4f}')
#         plt.xlabel('IoU Score')
#         plt.ylabel('Frequency')
#         plt.title('IoU Distribution')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Pixel accuracy distribution
#         plt.subplot(2, 2, 2)
#         plt.hist(all_pixel_accuracies, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
#         plt.axvline(avg_pixel_accuracy, color='r', linestyle='--', linewidth=2,
#                    label=f'Mean Accuracy: {avg_pixel_accuracy:.4f}')
#         plt.xlabel('Pixel Accuracy')
#         plt.ylabel('Frequency')
#         plt.title('Pixel Accuracy Distribution')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Per-class IoU comparison
#         plt.subplot(2, 2, 3)
#         class_mean_ious = [np.mean(all_class_ious[i]) if all_class_ious[i] else 0 
#                           for i in range(N_CLASSES)]
#         bars = plt.bar(CLASS_NAMES, class_mean_ious, alpha=0.7, color='coral', edgecolor='black')
#         plt.ylabel('Mean IoU')
#         plt.title('Per-Class IoU Performance')
#         plt.xticks(rotation=45)
#         plt.grid(True, alpha=0.3)
        
#         # Add value labels on bars
#         for bar, value in zip(bars, class_mean_ious):
#             plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                     f'{value:.3f}', ha='center', va='bottom')
        
#         # IoU vs Pixel Accuracy scatter
#         plt.subplot(2, 2, 4)
#         plt.scatter(all_ious, all_pixel_accuracies, alpha=0.6, color='purple')
#         plt.xlabel('IoU Score')
#         plt.ylabel('Pixel Accuracy')
#         plt.title('IoU vs Pixel Accuracy')
#         plt.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'comprehensive_metrics.png'), dpi=150, bbox_inches='tight')
#         print(f"Comprehensive metrics plot saved to {os.path.join(output_dir, 'comprehensive_metrics.png')}")
#         plt.close()

# if __name__ == "__main__":
#     main()
