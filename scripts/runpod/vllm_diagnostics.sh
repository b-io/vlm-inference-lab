#!/bin/bash
# Runpod Diagnostics Script
# Purpose: Check connectivity, SSH, and model serving status

# Colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}--- Runpod Diagnostics ---${NC}"

# 1. Environment Variables
echo -e "${YELLOW}[1/4] Environment Checks${NC}"
MODE=${RUNPOD_MODE:-existing}
echo "  RUNPOD_MODE     : $MODE"
echo "  RUNPOD_BASE_URL : $RUNPOD_BASE_URL"
echo "  MODEL_ID        : $MODEL_ID"

# 2. HTTP Connectivity & Model Status
echo -e "${YELLOW}[2/4] HTTP Connectivity${NC}"
if [ -n "$RUNPOD_BASE_URL" ]; then
    MODELS_URL="${RUNPOD_BASE_URL}/models"
    START_TIME=$(date +%s%N)
    if RESPONSE=$(curl -s -m 5 "$MODELS_URL"); then
        END_TIME=$(date +%s%N)
        LATENCY=$(( (END_TIME - START_TIME) / 1000000 ))
        echo -e "  /v1/models      : ${GREEN}OK (${LATENCY}ms)${NC}"
        
        MODELS=$(echo "$RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4 | tr '\n' ',' | sed 's/,$//')
        echo "  Models Found    : $MODELS"
        if echo "$MODELS" | grep -q "$MODEL_ID"; then
            echo -e "  Target Model    : ${GREEN}PRESENT${NC}"
        else
            echo -e "  Target Model    : ${RED}MISSING ($MODEL_ID)${NC}"
        fi
    else
        echo -e "  /v1/models      : ${RED}FAILED${NC}"
    fi
else
    echo "  RUNPOD_BASE_URL is not set."
fi

# 3. SSH Connectivity
echo -e "${YELLOW}[3/4] SSH Connectivity${NC}"
SSH_KEY="$RUNPOD_SSH_KEY"
REMOTE_HOST="$RUNPOD_SSH_HOST"
SSH_PORT="${RUNPOD_SSH_PORT:-22}"

if [ -n "$SSH_KEY" ] && [ -n "$REMOTE_HOST" ]; then
    echo -n "  Direct SSH ($REMOTE_HOST)..."
    if ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$REMOTE_HOST" "echo SSH_OK" &>/dev/null; then
        echo -e " ${GREEN}OK${NC}"
        
        echo -n "  SCP Support     ..."
        echo "SCP_TEST" > scp_test.tmp
        if scp -i "$SSH_KEY" -P "$SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no scp_test.tmp "${REMOTE_HOST}:/tmp/scp_test.tmp" &>/dev/null; then
            echo -e " ${GREEN}OK${NC}"
            ssh -i "$SSH_KEY" -p "$SSH_PORT" "$REMOTE_HOST" "rm /tmp/scp_test.tmp"
        else
            echo -e " ${YELLOW}FAILED${NC}"
        fi
        rm -f scp_test.tmp
    else
        echo -e " ${RED}FAILED${NC}"
    fi
else
    echo "  SSH details not fully provided in env."
fi

# 4. Managed Pod Heuristics
echo -e "${YELLOW}[4/4] Remote Environment Checks${NC}"
if [ -n "$SSH_KEY" ] && [ -n "$REMOTE_HOST" ]; then
    if ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ConnectTimeout=5 "$REMOTE_HOST" "pgrep -f 'vllm serve'" &>/dev/null; then
        echo -e "  vLLM Process    : ${YELLOW}DETECTED (Likely a managed service pod)${NC}"
        echo "  Recommendation  : Use RUNPOD_MODE=existing"
    else
        echo -e "  vLLM Process    : ${GREEN}NOT FOUND (Safe for generic deployment)${NC}"
    fi
else
    echo "  Skipping remote checks (no SSH access)."
fi

echo -e "${CYAN}--- Diagnostics Complete ---${NC}"
