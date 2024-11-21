#!/bin/bash
set -e

echo "Setting up datasets..."
CURRENT_DIR=$(pwd)

if ! command -v unzip &> /dev/null; then
    echo "Error: unzip is not installed"
    exit 1
fi

if [ ! -d "croco_dataset" ]; then
    echo "Extracting local zip files..."
    unzip -o croco_dataset.zip  
    rm croco_dataset.zip 
fi


if [ ! -d "raw_croco_d_images" ]; then
    echo "Starting download of croco_d_images..."
    echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
    source ~/.bashrc

    FILE_ID="1PgAz1ilCiWmDOP6bJaNm3hc8DiIjcQ4A"
    OUTPUT_FILE="croco_d_images.zip"

    gdown "1PgAz1ilCiWmDOP6bJaNm3hc8DiIjcQ4A"

    echo "Verifying downloaded file..."
    if file "${OUTPUT_FILE}" | grep -q 'Zip archive data'; then
        echo "File verified as a zip archive."
        unzip -o croco_d_images.zip -d raw_croco_d_images
        rm croco_d_images.zip
    else
        echo "Error: Downloaded file is not a valid zip archive."
        file "${OUTPUT_FILE}"  # Show the actual type for debugging
        exit 1
    fi
fi

if [ ! -d "vg_relation_images" ]; then
    echo "Downloading VG Relation dataset..."
    mkdir -p vg_relation_images
    echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
    source ~/.bashrc

    FILE_ID="1tsGyGcfMCUUTkicTlSBrQtWR81nlQUhG"
    ZIPPED_FILE="vg_relation_images.zip"

    gdown "1tsGyGcfMCUUTkicTlSBrQtWR81nlQUhG"

    echo "Verifying downloaded file..."
    if file "${ZIPPED_FILE}" | grep -q 'Zip archive data'; then
        echo "File verified as a zip archive."
        unzip -j -o "${ZIPPED_FILE}" -d vg_relation_images
        rm "${ZIPPED_FILE}"
    else
        echo "Error: Downloaded file is not a valid zip archive."
        file "${ZIPPED_FILE}"  # Show the actual type for debugging
        exit 1
    fi
fi
 
# Download and extract COCO dataset if not already present
if [ ! -d "coco_images/train2017" ] || [ ! -d "coco_images/val2017" ]; then
    echo "Downloading COCO dataset..."
    mkdir -p coco_images

    if [ ! -f "coco_images/train2017.zip" ] || [ ! -f "coco_images/val2017.zip" ]; then
        wget -P coco_images http://images.cocodataset.org/zips/train2017.zip
        wget -P coco_images http://images.cocodataset.org/zips/val2017.zip
    fi

    unzip coco_images/train2017.zip -d coco_images
    unzip coco_images/val2017.zip -d coco_images

    rm coco_images/train2017.zip coco_images/val2017.zip
fi


echo "calling script to create dataset splits"
/usr/bin/env python scripts/fill_datasets.py

echo "Dataset setup complete!"