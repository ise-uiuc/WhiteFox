#!/bin/bash

# WhiteFox Ollama Integration Script
# This script helps set up and run Ollama with WhiteFox

set -e

OLLAMA_MODEL="starcoder"
OLLAMA_URL="http://localhost:11434"
PROMPT_DIR="Prompts"
OUTPUT_DIR="ollama-generated"
NUM_GENERATIONS=10
TEMPERATURE=1.0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup     - Install and setup Ollama"
    echo "  check     - Check Ollama connection and available models"
    echo "  generate  - Generate code using Ollama"
    echo "  service   - Run Ollama service (continuous monitoring)"
    echo ""
    echo "Options:"
    echo "  --model MODEL          Ollama model to use (default: codellama:7b)"
    echo "  --url URL              Ollama server URL (default: http://localhost:11434)"
    echo "  --prompt-dir DIR       Directory containing prompts (default: Prompts)"
    echo "  --output-dir DIR       Output directory (default: ollama-generated)"
    echo "  --num N                Number of generations per prompt (default: 10)"
    echo "  --temperature T        Temperature for generation (default: 1.0)"
    echo ""
    echo "Examples:"
    echo "  $0 setup --model codellama:13b"
    echo "  $0 generate --prompt-dir Prompts/torch-inductor/step0"
    echo "  $0 service --model mistral:7b --num 5"
}

check_ollama_installed() {
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}Ollama is not installed.${NC}"
        return 1
    fi
    return 0
}

install_ollama() {
    echo -e "${YELLOW}Installing Ollama...${NC}"
    curl -fsSL https://ollama.ai/install.sh | sh
    echo -e "${GREEN}Ollama installed successfully!${NC}"
}

check_ollama_running() {
    if ! curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo -e "${RED}Ollama server is not running at $OLLAMA_URL${NC}"
        echo "Start Ollama with: ollama serve"
        return 1
    fi
    return 0
}

pull_model() {
    local model=$1
    echo -e "${YELLOW}Pulling model $model...${NC}"
    ollama pull "$model"
    echo -e "${GREEN}Model $model pulled successfully!${NC}"
}

check_model_available() {
    local model=$1
    if ! ollama list | grep -q "$model"; then
        echo -e "${YELLOW}Model $model is not available locally.${NC}"
        read -p "Do you want to pull it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pull_model "$model"
        else
            echo -e "${RED}Cannot proceed without the model.${NC}"
            exit 1
        fi
    fi
}

setup_ollama() {
    echo -e "${YELLOW}Setting up Ollama for WhiteFox...${NC}"
    
    if ! check_ollama_installed; then
        install_ollama
    fi
    
    if ! check_ollama_running; then
        echo -e "${YELLOW}Starting Ollama service...${NC}"
        ollama serve &
        sleep 5
    fi
    
    pull_model "$OLLAMA_MODEL"
    
    echo -e "${GREEN}Ollama setup completed!${NC}"
}

check_ollama() {
    echo -e "${YELLOW}Checking Ollama status...${NC}"
    
    if ! check_ollama_installed; then
        echo -e "${RED}Run '$0 setup' to install Ollama${NC}"
        exit 1
    fi
    
    if ! check_ollama_running; then
        echo -e "${RED}Start Ollama with: ollama serve${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Ollama is running at $OLLAMA_URL${NC}"
    echo ""
    echo "Available models:"
    ollama list
}

run_generation() {
    echo -e "${YELLOW}Running Ollama generation...${NC}"
    
    if ! check_ollama_running; then
        echo -e "${RED}Ollama server is not running. Start it with: ollama serve${NC}"
        exit 1
    fi
    
    check_model_available "$OLLAMA_MODEL"
    
    python ollama_gen.py \
        --prompt-dir "$PROMPT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --model "$OLLAMA_MODEL" \
        --base-url "$OLLAMA_URL" \
        --num "$NUM_GENERATIONS" \
        --temperature "$TEMPERATURE"
}

run_service() {
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    
    if ! check_ollama_running; then
        echo -e "${RED}Ollama server is not running. Start it with: ollama serve${NC}"
        exit 1
    fi
    
    check_model_available "$OLLAMA_MODEL"
    
    python ollama_service.py \
        --prompt-dir "$PROMPT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --model "$OLLAMA_MODEL" \
        --base-url "$OLLAMA_URL" \
        --num "$NUM_GENERATIONS" \
        --temperature "$TEMPERATURE"
}

COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        setup|check|generate|service)
            COMMAND="$1"
            shift
            ;;
        --model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --url)
            OLLAMA_URL="$2"
            shift 2
            ;;
        --prompt-dir)
            PROMPT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

case "$COMMAND" in
    setup)
        setup_ollama
        ;;
    check)
        check_ollama
        ;;
    generate)
        run_generation
        ;;
    service)
        run_service
        ;;
    *)
        echo -e "${RED}No command specified.${NC}"
        print_usage
        exit 1
        ;;
esac
