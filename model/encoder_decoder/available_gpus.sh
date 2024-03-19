# List available GPU types in a specific region
gcloud compute accelerator-types list --filter="zone:(us-central1-a)" --format="value(name)"

# Hypothetical script logic to select a GPU (pseudo-code)
available_gpus = ["nvidia-tesla-k80", "nvidia-tesla-p100", ...] # Result from the above command
preferred_gpu = "nvidia-tesla-t4" # Example preference

if preferred_gpu in available_gpus:
    selected_gpu = preferred_gpu
else:
    selected_gpu = available_gpus[0] # Fallback to the first available GPU
