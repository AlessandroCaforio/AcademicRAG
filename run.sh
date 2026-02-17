#!/bin/bash
# AcademicRAG — Quick Start

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}AcademicRAG${NC}"
echo "=========================="

# Virtual environment
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

# Dependencies
if [ ! -f "venv/.installed" ]; then
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check Ollama (optional — only needed for Ollama mode)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${BLUE}Ollama not running. Start it with: ollama serve${NC}"
    echo "You can still use Claude Code or Claude API mode."
fi

# Launch
echo -e "${GREEN}Starting AcademicRAG...${NC}"
echo "Open http://localhost:8501 in your browser"
echo ""

streamlit run app/main.py --server.headless true
