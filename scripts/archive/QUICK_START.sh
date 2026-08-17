#!/bin/bash
# Dish-Chat Quick Start Guide
# Interactive deployment helper
# Created: 2026-02-27

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

clear

echo -e "${BLUE}"
echo "========================================"
echo "  Dish-Chat Deployment Quick Start"
echo "========================================"
echo -e "${NC}"
echo ""
echo "Source: montjac@10.79.85.47:~/dish-chat"
echo "Target: jakebot@10.79.85.35:~/Jakes-agent"
echo ""
echo "This wizard will guide you through the deployment process."
echo ""

# Step 1: Transfer
echo -e "${GREEN}[Step 1/4] File Transfer${NC}"
echo ""
echo "This step transfers files from 3080 to 3090."
echo "Required: SSH access to both servers"
echo ""
read -p "Ready to transfer files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "./transfer-dishchat.sh" ]; then
        echo "Starting transfer..."
        bash ./transfer-dishchat.sh
        echo ""
        echo -e "${GREEN}✓ Transfer complete${NC}"
    else
        echo -e "${YELLOW}⚠ transfer-dishchat.sh not found in current directory${NC}"
        echo "Please run this script from the deployment package directory"
        exit 1
    fi
else
    echo "Skipping transfer step"
fi

echo ""
echo -e "${GREEN}[Step 2/4] Copy Management Scripts${NC}"
echo ""
echo "Next, copy the management scripts to the target server:"
echo ""
echo "Run these commands:"
echo -e "${BLUE}"
cat << 'COPY_CMDS'
scp setup-dishchat.sh jakebot@10.79.85.35:~/Jakes-agent/
scp start-dishchat.sh jakebot@10.79.85.35:~/Jakes-agent/
scp stop-dishchat.sh jakebot@10.79.85.35:~/Jakes-agent/
scp restart-dishchat.sh jakebot@10.79.85.35:~/Jakes-agent/
scp verify-deployment.sh jakebot@10.79.85.35:~/Jakes-agent/
scp DEPLOYMENT_README.md jakebot@10.79.85.35:~/Jakes-agent/
COPY_CMDS
echo -e "${NC}"
echo ""
read -p "Press Enter when scripts are copied..."

echo ""
echo -e "${GREEN}[Step 3/4] Run Setup on Target${NC}"
echo ""
echo "SSH into the target server and run setup:"
echo ""
echo -e "${BLUE}"
cat << 'SETUP_CMDS'
ssh jakebot@10.79.85.35
cd ~/Jakes-agent
chmod +x *.sh
bash setup-dishchat.sh
SETUP_CMDS
echo -e "${NC}"
echo ""
echo "This will take 5-10 minutes."
echo ""
read -p "Press Enter when setup is complete..."

echo ""
echo -e "${GREEN}[Step 4/4] Start and Verify${NC}"
echo ""
echo "On the target server, run:"
echo ""
echo -e "${BLUE}"
cat << 'START_CMDS'
bash start-dishchat.sh
bash verify-deployment.sh
START_CMDS
echo -e "${NC}"
echo ""
read -p "Press Enter when services are started..."

echo ""
echo -e "${GREEN}"
echo "========================================"
echo "  Deployment Process Complete!"
echo "========================================"
echo -e "${NC}"
echo ""
echo "Verify the deployment with:"
echo "  curl http://10.79.85.35:8000/health"
echo "  http://10.79.85.35:8000/docs"
echo ""
echo "View logs:"
echo "  ssh jakebot@10.79.85.35 'tail -f ~/Jakes-agent/logs/backend.log'"
echo ""
echo "Service management:"
echo "  bash start-dishchat.sh   # Start services"
echo "  bash stop-dishchat.sh    # Stop services"
echo "  bash restart-dishchat.sh # Restart services"
echo ""
echo "For more details, see:"
echo "  DEPLOYMENT_README.md"
echo "  EXECUTION_SUMMARY.md"
echo ""

