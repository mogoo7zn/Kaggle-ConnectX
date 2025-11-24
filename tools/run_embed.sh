#!/bin/bash

echo "========================================"
echo "Embedding Model into main.py"
echo "========================================"
echo

echo "Step 1: Converting model to Base64..."
python embed_model.py ../submission/best_model.pth model_weights_embedded.txt
if [ $? -ne 0 ]; then
    echo "Error in step 1!"
    exit 1
fi
echo

echo "Step 2: Creating embedded main.py..."
python create_embedded_main.py --original ../submission/main_backup.py --model model_weights_embedded.txt --output ../submission/main.py
if [ $? -ne 0 ]; then
    echo "Error in step 2!"
    exit 1
fi
echo

echo "Step 3: Cleaning up..."
rm model_weights_embedded.txt
echo

echo "========================================"
echo "Completed! main.py is ready!"
echo "========================================"

