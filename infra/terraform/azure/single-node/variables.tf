variable "project_name" {
  type        = string
  default     = "vlm-lab"
  description = "Project name used for resource naming"
}

variable "location" {
  type        = string
  default     = "eastus"
  description = "Azure region for deployment"
}

variable "vm_size" {
  type        = string
  default     = "Standard_ND96isr_H100_v5"
  description = "Azure VM size for GPU node"
}

variable "admin_username" {
  type        = string
  default     = "azureuser"
  description = "Admin username for the VM"
}

variable "ssh_public_key_path" {
  type        = string
  default     = "~/.ssh/id_rsa.pub"
  description = "Path to the SSH public key for VM access"
}

variable "allowed_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "Allowed CIDR for SSH and API access"
}

variable "hf_token" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Hugging Face token for model downloads"
}
