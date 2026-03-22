# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$LOCATION = if ($args[0]) { $args[0] } else { "eastus" }
$SKU = if ($args[1]) { $args[1] } else { "Standard_ND96isr_H100_v5" }

Write-Host "Checking availability of $SKU in $LOCATION..."
az vm list-skus --location $LOCATION --size $SKU --out table
