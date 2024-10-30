import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, GlobalAveragePooling2D, 
    Concatenate, Multiply, Add, LayerNormalization
)

def load_unet_model(model_path):
    """
    Load the saved U-Net model and create feature extractor
    """
    # Load the saved model with custom objects if needed
    custom_objects = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m
    }
    
    unet_model = load_model(model_path, custom_objects=custom_objects)
    
    # Create feature extractor model (getting c11 layer output)
    # Find the c11 layer (third to last layer, before final Conv2D)
    c11_layer_name = unet_model.layers[-3].name
    feature_extractor = Model(
        inputs=unet_model.input,
        outputs=unet_model.get_layer(c11_layer_name).output
    )
    
    return feature_extractor

class MultimodalFusion(tf.keras.Model):
    def __init__(self, n_classes=3, fusion_type='multiplicative', fusion_dim=256):
        super(MultimodalFusion, self).__init__()
        self.fusion_type = fusion_type
        self.n_classes = n_classes
        self.fusion_dim = fusion_dim
        
        # LSTM projection
        self.lstm_projection = Dense(fusion_dim, activation='relu')
        
        # U-Net feature processing
        self.unet_gap = GlobalAveragePooling2D()
        self.unet_projection = Dense(fusion_dim, activation='relu')
        
        # Layers for different fusion types
        if fusion_type == 'nonlinear':
            self.fusion_layers = [
                Dense(fusion_dim, activation='relu'),
                LayerNormalization(),
                Dense(fusion_dim // 2, activation='relu'),
                LayerNormalization()
            ]
        elif fusion_type == 'gated':
            self.lstm_gate = Dense(fusion_dim, activation='sigmoid')
            self.unet_gate = Dense(fusion_dim, activation='sigmoid')
            
        # Final classification layers
        self.final_dense = Dense(fusion_dim // 2, activation='relu')
        self.classifier = Dense(n_classes, activation='softmax')
    
    # ... [rest of the MultimodalFusion class implementation remains the same]

def create_fused_model(lstm_model, unet_model_path, fusion_type='multiplicative'):
    """
    Create fused model using LSTM model and path to saved U-Net model
    
    Parameters:
    -----------
    lstm_model : keras.Model
        The LSTM model instance
    unet_model_path : str
        Path to the saved U-Net model (.h5 or SavedModel format)
    fusion_type : str
        Type of fusion to use ('multiplicative', 'additive', 'nonlinear', or 'gated')
    """
    # Extract features from LSTM (before final dense layer)
    lstm_feature_model = Model(
        inputs=lstm_model.input,
        outputs=lstm_model.layers[-3].output
    )
    
    # Load and create U-Net feature extractor
    unet_feature_model = load_unet_model(unet_model_path)
    
    # Create fusion model
    fusion_model = MultimodalFusion(n_classes=3, fusion_type=fusion_type)
    
    # Define inputs
    lstm_input = Input(shape=(5, 4))  # LSTM input shape
    unet_input = Input(shape=(256, 256, 1))  # U-Net input shape
    
    # Get features
    lstm_features = lstm_feature_model(lstm_input)
    unet_features = unet_feature_model(unet_input)
    
    # Apply fusion
    outputs = fusion_model([lstm_features, unet_features])
    
    # Create combined model
    combined_model = Model(
        inputs=[lstm_input, unet_input],
        outputs=outputs
    )
    
    return combined_model

# Example usage
def example_usage():
    # Path to your saved U-Net model
    unet_model_path = 'path/to/your/unet_model.h5'
    
    # Create fused model
    fusion_types = ['multiplicative', 'additive', 'nonlinear', 'gated']
    
    for fusion_type in fusion_types:
        try:
            combined_model = create_fused_model(
                lstm_model=lstm_model,  # Your LSTM model
                unet_model_path=unet_model_path,
                fusion_type=fusion_type
            )
            
            # Compile with same metrics
            combined_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy', f1_m, precision_m, recall_m]
            )
            
            print(f"\nCreated {fusion_type} fusion model")
            combined_model.summary()
            
        except Exception as e:
            print(f"Error creating {fusion_type} fusion model: {str(e)}")

# Usage example:
def main():
    # Assume you have your LSTM model already created
    # lstm_model = ... (your LSTM model)
    
    # Path to your saved U-Net model
    unet_model_path = 'path/to/saved/unet_model.h5'
    
    # Create the fused model
    combined_model = create_fused_model(
        lstm_model=lstm_model,
        unet_model_path=unet_model_path,
        fusion_type='multiplicative'
    )
    
    # Now you can use the combined_model for training or inference
    # combined_model.fit([lstm_data, unet_data], labels, ...)