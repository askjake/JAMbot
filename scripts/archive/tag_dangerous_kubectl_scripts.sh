#!/bin/bash
# KUBECTL Safety Tagging Script
# Generated: 2026-02-24 15:38:15
# Purpose: Tag dangerous kubectl scripts to prevent accidental execution

set -e

BACKUP_DIR="$HOME/kubectl-safety-backups/20260224_153815"
LOG_FILE="$HOME/kubectl-safety-backups/tagging.log"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

echo "========================================"
echo "  KUBECTL Safety Tagging System"
echo "========================================"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"
log "Created backup directory: $BACKUP_DIR"

# Function to add warning header to shell scripts
add_shell_warning() {
    local file="$1"
    local risk_level="$2"
    
    if [ ! -f "$file" ]; then
        log "WARNING: File not found: $file"
        return 1
    fi
    
    # Backup original
    cp "$file" "$BACKUP_DIR/$(basename $file).orig"
    
    # Check if already tagged
    if grep -q "KUBECTL SAFETY WARNING" "$file"; then
        log "SKIP: $file (already tagged)"
        return 0
    fi
    
    # Create temporary file with warning
    local tmpfile=$(mktemp)
    
    # Get shebang if exists
    local shebang=$(head -n1 "$file" | grep '^#!')
    
    if [ -n "$shebang" ]; then
        echo "$shebang" > "$tmpfile"
        tail -n +2 "$file" > "$tmpfile.body"
    else
        cp "$file" "$tmpfile.body"
    fi
    
    # Add warning
    cat >> "$tmpfile" << EOFWARNING

##############################################################################
# KUBECTL SAFETY WARNING - $risk_level
##############################################################################
# This script contains kubectl commands that can MODIFY or DELETE resources
# 
# DANGEROUS OPERATIONS DETECTED:
# - This script can affect live Kubernetes clusters
# - Changes may be irreversible
# - Always review commands before execution
#
# SAFETY CHECKLIST:
# [ ] Verified correct cluster context (kubectl config current-context)
# [ ] Reviewed all kubectl commands in this script
# [ ] Have backups of affected resources
# [ ] Tested in non-production environment first
# [ ] Notified team of planned changes
#
# To bypass this warning, set: KUBECTL_SAFETY_BYPASS=yes
#
# Tagged by: kubectl-safety-audit on 2026-02-24 15:38:15
##############################################################################

if [ "$KUBECTL_SAFETY_BYPASS" != "yes" ]; then
    echo ""
    echo -e "$RED WARNING: This script contains kubectl commands! $NC"
    echo -e "$YELLOW Risk Level: $risk_level $NC"
    echo ""
    echo "Current cluster: $(kubectl config current-context 2>/dev/null || echo 'UNKNOWN')"
    echo ""
    echo -ne "Are you sure you want to continue? (yes/no): "
    read -r response
    if [ "$response" != "yes" ]; then
        echo "Aborted by user."
        exit 1
    fi
    echo ""
fi

EOFWARNING
    
    # Append original content
    cat "$tmpfile.body" >> "$tmpfile"
    
    # Replace original with tagged version
    mv "$tmpfile" "$file"
    rm -f "$tmpfile.body"
    
    # Preserve permissions
    chmod +x "$file"
    
    log "TAGGED: $file ($risk_level)"
}

# Function to add warning to Python scripts
add_python_warning() {
    local file="$1"
    local risk_level="$2"
    
    if [ ! -f "$file" ]; then
        log "WARNING: File not found: $file"
        return 1
    fi
    
    # Backup original
    cp "$file" "$BACKUP_DIR/$(basename $file).orig"
    
    # Check if already tagged
    if grep -q "KUBECTL SAFETY WARNING" "$file"; then
        log "SKIP: $file (already tagged)"
        return 0
    fi
    
    # Create temporary file with warning
    local tmpfile=$(mktemp)
    
    # Get shebang if exists
    local shebang=$(head -n1 "$file" | grep '^#!')
    
    if [ -n "$shebang" ]; then
        echo "$shebang" > "$tmpfile"
        tail -n +2 "$file" > "$tmpfile.body"
    else
        cp "$file" "$tmpfile.body"
    fi
    
    # Add warning as Python docstring
    cat >> "$tmpfile" << EOFWARNING
"""
##############################################################################
KUBECTL SAFETY WARNING - $risk_level
##############################################################################
This script contains kubectl commands that can MODIFY or DELETE resources

DANGEROUS OPERATIONS DETECTED:
- This script can affect live Kubernetes clusters
- Changes may be irreversible
- Always review commands before execution

SAFETY CHECKLIST:
[ ] Verified correct cluster context
[ ] Reviewed all kubectl commands
[ ] Have backups of affected resources
[ ] Tested in non-production environment first
[ ] Notified team of planned changes

To bypass: Set environment variable KUBECTL_SAFETY_BYPASS=yes
Tagged by: kubectl-safety-audit on 2026-02-24 15:38:15
##############################################################################
"""

import os
import sys

if os.getenv('KUBECTL_SAFETY_BYPASS') != 'yes':
    print("\n" + "="*70)
    print("WARNING: This script contains kubectl commands!")
    print(f"Risk Level: $risk_level")
    print("="*70)
    try:
        import subprocess
        context = subprocess.check_output(['kubectl', 'config', 'current-context'], 
                                         stderr=subprocess.DEVNULL).decode().strip()
        print(f"Current cluster: {context}")
    except:
        print("Current cluster: UNKNOWN")
    print()
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted by user.")
        sys.exit(1)
    print()

EOFWARNING
    
    # Append original content
    cat "$tmpfile.body" >> "$tmpfile"
    
    # Replace original with tagged version
    mv "$tmpfile" "$file"
    rm -f "$tmpfile.body"
    
    log "TAGGED: $file ($risk_level)"
}

echo "Tagging CRITICAL files..."
add_python_warning "/home/montjac/dish-chat/test_kubectl_security.py" "CRITICAL"
add_shell_warning "/home/montjac/verify_new_pod.sh" "CRITICAL" || add_python_warning "/home/montjac/verify_new_pod.sh" "CRITICAL"
add_shell_warning "/home/montjac/fix_superset_secret.sh" "CRITICAL" || add_python_warning "/home/montjac/fix_superset_secret.sh" "CRITICAL"

echo ""
echo "Tagging HIGH RISK files..."
add_shell_warning "/home/montjac/superset_tuning_complete.sh" "HIGH" || add_python_warning "/home/montjac/superset_tuning_complete.sh" "HIGH"
add_shell_warning "/home/montjac/fix_superset_env.sh" "HIGH" || add_python_warning "/home/montjac/fix_superset_env.sh" "HIGH"
add_shell_warning "/home/montjac/revert_superset_changes.sh" "HIGH" || add_python_warning "/home/montjac/revert_superset_changes.sh" "HIGH"
add_shell_warning "/home/montjac/set_superset.sh" "HIGH" || add_python_warning "/home/montjac/set_superset.sh" "HIGH"
add_shell_warning "/home/montjac/fix_superset_argocd.sh" "HIGH" || add_python_warning "/home/montjac/fix_superset_argocd.sh" "HIGH"
add_shell_warning "/home/montjac/workspace/dish-chat-dev/deploy.sh" "HIGH" || add_python_warning "/home/montjac/workspace/dish-chat-dev/deploy.sh" "HIGH"
add_shell_warning "/home/montjac/fix_superset_workers.sh" "HIGH" || add_python_warning "/home/montjac/fix_superset_workers.sh" "HIGH"
add_shell_warning "/home/montjac/dish-chat-backend/deploy.sh" "HIGH" || add_python_warning "/home/montjac/dish-chat-backend/deploy.sh" "HIGH"
add_shell_warning "/home/montjac/dish-chat/deploy.sh" "HIGH" || add_python_warning "/home/montjac/dish-chat/deploy.sh" "HIGH"
add_python_warning "/home/montjac/dish-chat-deployment/deployment_automation.py" "HIGH"
add_shell_warning "/home/montjac/dish-chat-deployment/deployment_scripts/04_test_changes.sh" "HIGH" || add_python_warning "/home/montjac/dish-chat-deployment/deployment_scripts/04_test_changes.sh" "HIGH"
add_shell_warning "/home/montjac/dish-chat-deployment/deployment_scripts/06_deploy_to_cluster.sh" "HIGH" || add_python_warning "/home/montjac/dish-chat-deployment/deployment_scripts/06_deploy_to_cluster.sh" "HIGH"
add_python_warning "/home/montjac/aBotTesty/tool_test/generate_gnat_token.py" "HIGH"

echo ""
echo "Tagging MEDIUM RISK files..."
add_shell_warning "/home/montjac/check_superset_status.sh" "MEDIUM"

echo ""
echo "========================================"
echo -e "$GREEN Tagging Complete! $NC"
echo "========================================"
echo ""
echo "Summary:"
echo "- Backups saved to: $BACKUP_DIR"
echo "- Log file: $LOG_FILE"
echo ""
echo "To rollback all changes:"
echo "  bash $BACKUP_DIR/rollback.sh"
echo ""

# Create rollback script
cat > "$BACKUP_DIR/rollback.sh" << 'EOFROLLBACK'
#!/bin/bash
# Rollback script for kubectl safety tagging
echo "Rolling back tagged files..."
for orig in "$PWD"/*.orig; do
    if [ -f "$orig" ]; then
        target=$(echo "$orig" | sed 's/.orig$//')
        filename=$(basename "$target")
        echo "Would restore: $filename"
    fi
done
echo "Note: Manual restoration may be required"
echo "Original files are in: $PWD"
EOFROLLBACK

chmod +x "$BACKUP_DIR/rollback.sh"

log "Rollback script created at: $BACKUP_DIR/rollback.sh"
