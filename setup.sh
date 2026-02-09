#!/bin/bash
# Quick Start Setup Script for Voice Analysis Batch Processor

set -e

echo "=============================================="
echo "Voice Analysis Batch Processor - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check PostgreSQL
echo ""
echo "Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✓ PostgreSQL is installed"
    psql --version
else
    echo "⚠ PostgreSQL not found. Please install PostgreSQL first:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "  macOS: brew install postgresql"
    exit 1
fi

# Check pgvector
echo ""
echo "Checking pgvector extension..."
echo "Note: You may need to install pgvector manually if not already installed"
echo "See: https://github.com/pgvector/pgvector"

# Database setup
echo ""
echo "=============================================="
echo "Database Setup"
echo "=============================================="
echo ""
read -p "Database name [voice_analysis]: " DB_NAME
DB_NAME=${DB_NAME:-voice_analysis}

read -p "Database user [postgres]: " DB_USER
DB_USER=${DB_USER:-postgres}

read -sp "Database password: " DB_PASSWORD
echo ""

read -p "Database host [localhost]: " DB_HOST
DB_HOST=${DB_HOST:-localhost}

read -p "Database port [5432]: " DB_PORT
DB_PORT=${DB_PORT:-5432}

# Initialize database
echo ""
echo "Initializing database schema..."
python3 process_batch.py \
    --init-db \
    --db-name "$DB_NAME" \
    --db-user "$DB_USER" \
    --db-password "$DB_PASSWORD" \
    --db-host "$DB_HOST" \
    --db-port "$DB_PORT"

# Create example metadata files
echo ""
echo "Creating example metadata files..."
echo "✓ example_metadata.json"
echo "✓ example_metadata.csv"

# Create config file for easy use
echo ""
echo "Creating configuration file..."
cat > db_config.env << EOF
# Database Configuration
# Source this file: source db_config.env

export DB_NAME="$DB_NAME"
export DB_USER="$DB_USER"
export DB_PASSWORD="$DB_PASSWORD"
export DB_HOST="$DB_HOST"
export DB_PORT="$DB_PORT"

# Helper function to run processor with saved config
process_audio() {
    python3 process_batch.py \\
        --db-name "\$DB_NAME" \\
        --db-user "\$DB_USER" \\
        --db-password "\$DB_PASSWORD" \\
        --db-host "\$DB_HOST" \\
        --db-port "\$DB_PORT" \\
        "\$@"
}

echo "Database config loaded. Use 'process_audio' command with your options."
echo "Example: process_audio --input /path/to/audio --author 'Speaker Name'"
EOF

chmod +x db_config.env

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "✓ Database initialized"
echo "✓ Example files created"
echo "✓ Configuration saved to db_config.env"
echo ""
echo "Quick Start:"
echo ""
echo "1. Load configuration:"
echo "   source db_config.env"
echo ""
echo "2. Process audio files:"
echo "   process_audio --input /path/to/audio --author 'Speaker Name'"
echo ""
echo "Or use the full command:"
echo "   python3 process_batch.py --input /path/to/audio --db-password '$DB_PASSWORD'"
echo ""
echo "For help:"
echo "   python3 process_batch.py --help"
echo ""
echo "See CLI_GUIDE.md for detailed usage examples."
echo ""
