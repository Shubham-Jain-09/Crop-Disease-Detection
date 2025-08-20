import h5py
import os
import shutil

def fix_model_layer_names(input_path, output_path):
    """
    Fix model layer names by replacing forward slashes with underscores
    """
    try:
        with h5py.File(input_path, 'r') as src_file:
            with h5py.File(output_path, 'w') as dst_file:
                def copy_group(src, dst, path=''):
                    for key in src.keys():
                        if isinstance(src[key], h5py.Group):
                            # Create new group with sanitized name
                            new_key = key.replace('/', '_')
                            new_group = dst.create_group(new_key)
                            copy_group(src[key], new_group, path + '/' + new_key)
                        else:
                            # Copy dataset
                            dst.create_dataset(key, data=src[key][:])
                
                copy_group(src_file, dst_file)
        
        print(f"✓ Model fixed and saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error fixing model: {e}")
        return False

def backup_and_fix_model(model_path):
    """
    Create a backup and fix the model
    """
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    # Create backup
    backup_path = model_path + '.backup'
    shutil.copy2(model_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    
    # Create fixed version
    fixed_path = model_path.replace('.h5', '_fixed.h5')
    success = fix_model_layer_names(model_path, fixed_path)
    
    if success:
        # Replace original with fixed version
        shutil.move(fixed_path, model_path)
        print(f"✓ Original model replaced with fixed version")
        return True
    else:
        print(f"✗ Failed to fix model, keeping original")
        return False

if __name__ == "__main__":
    # Fix the problematic models
    models_to_fix = [
        'model/DenseNet121v2_95.h5',
        'model/SoilNet_93_86.h5'
    ]
    
    for model_path in models_to_fix:
        print(f"\nProcessing: {model_path}")
        if os.path.exists(model_path):
            backup_and_fix_model(model_path)
        else:
            print(f"✗ Model file not found: {model_path}")
    
    print("\nModel fixing completed!")
