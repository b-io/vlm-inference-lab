#!/bin/bash
LOCATION=${1:-"eastus"}
SKU=${2:-"Standard_ND96isr_H100_v5"}

echo "Checking availability of $SKU in $LOCATION..."
az vm list-skus --location $LOCATION --size $SKU --out table
